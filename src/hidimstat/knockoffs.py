import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_memory

from hidimstat._utils.docstring import _aggregate_docstring
from hidimstat.base_variable_importance import BaseVariableImportance
from hidimstat.statistical_tools.aggregation import quantile_aggregation
from hidimstat.statistical_tools.gaussian_knockoffs import GaussianKnockoffs
from hidimstat.statistical_tools.lasso_test import lasso_statistic_with_sampling
from hidimstat.statistical_tools.multiple_testing import fdr_threshold


class ModelXKnockoff(BaseVariableImportance):
    """
    Model-X Knockoff

    This module implements the Model-X knockoff inference procedure, which is an approach
    to control the False Discovery Rate (FDR) based on :footcite:t:`candes2018panning`.
    The original implementation can be found at
    https://github.com/msesia/knockoff-filter/blob/master/R/knockoff/R/knockoff_filter.R
    The noisy variables are generated with second-order knockoff variables using the equi-correlated method.

    In addition, this function generates multiple sets of Gaussian knockoff variables and calculates
    the test statistics for each set. It then aggregates the test statistics across
    the sets to improve stability and power.

    Parameters
    ----------
    X : 2D ndarray (n_samples, n_features)
        The design matrix.

    y : 1D ndarray (n_samples, )
        The target vector.

    estimator : sklearn estimator instance or a cross validation instance
        The estimator used for fitting the data and computing the test statistics.
        This can be any estimator with a `fit` method that accepts a 2D array and
        a 1D array, and a `coef_` attribute that returns a 1D array of coefficients.
        Examples include LassoCV, LogisticRegressionCV, and LinearRegression.

        *Configuration example:*

        | LassoCV(alphas=alphas, n_jobs=None, verbose=0, max_iter=1000,
        | cv=KFold(n_splits=5, shuffle=True, random_state=0), tol=1e-8)

        | LogisticRegressionCV(penalty="l1", max_iter=1000, solver="liblinear",
        | cv=KFold(n_splits=5, shuffle=True, random_state=0), n_jobs=None,
        | tol=1e-8)

        | LogisticRegressionCV(penalty="l2", max_iter=1000, n_jobs=None,
        | verbose=0, cv=KFold(n_splits=5, shuffle=True, random_state=0),
        | tol=1e-8,)

    preconfigure_estimator : callable, default=preconfigure_estimator_LassoCV
        A function that configures the estimator for the Model-X knockoff procedure.
        If provided, this function will be called with the estimator, X, X_tilde, and y
        as arguments, and should modify the estimator in-place.

    fdr : float, default=0.1
        The desired controlled False Discovery Rate (FDR) level.

    centered : bool, default=True
        Whether to standardize the data before performing the inference procedure.

    cov_estimator : estimator object, default=LedoitWolf()
        Estimator for empirical covariance matrix.

    joblib_verbose : int, default=0
        Verbosity level for parallel jobs.

    n_bootstraps : int, default=1
        Number of bootstrap samples for aggregation.

    n_jobs : int, default=1
        Number of parallel jobs.

    random_state : int or None, default=None
        The random seed used to generate the Gaussian knockoff variables.

    tol_gauss : float, default=1e-14
        A tolerance value used for numerical stability in the calculation of
        the Cholesky decomposition in the gaussian generation function.

    memory : str or Memory object, default=None
        Used to cache the output of the clustering and inference computation.
        By default, no caching is done. If provided, it should be the path
        to the caching directory or a joblib.Memory object.

    Returns
    -------
    selected : ndarray or list of ndarrays
        Selected feature indices. List if n_bootstraps>1.

    test_scores : ndarray or list of ndarrays
        Test statistics. List if n_bootstraps>1.

    threshold : float or list of floats
        Knockoff thresholds. List if n_bootstraps>1.

    X_tildes : ndarray or list of ndarrays
        Generated knockoff variables. List if n_bootstraps>1.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        ko_generator=GaussianKnockoffs(),
        statistical_test=lasso_statistic_with_sampling,
        n_repeat=1,
        centered=True,
        random_state=None,
        joblib_verbose=0,
        memory=None,
        n_jobs=1,
    ):
        super().__init__()
        self.generator = ko_generator
        self.statistical_test = statistical_test
        assert n_repeat > 0, "n_samplings must be positive"
        self.n_repeat = n_repeat
        self.centered = centered

        self.randoms_state = random_state
        self.memory = check_memory(memory)
        self.joblib_verbose = joblib_verbose
        # unnecessary to have n_jobs > number of bootstraps
        self.n_jobs = min(n_repeat, n_jobs)

        self.test_scores_ = None
        self.threshold_fdr_ = None
        self.aggregated_eval_ = None
        self.aggregated_pval_ = None

    def fit(self, X, y=None):
        """
        Fit the Model-X Knockoff model by training the generator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data matrix where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,), default=None
            Target values. Not used in this method.

        Returns
        -------
        self : object
            Returns the instance itself.

        Notes
        -----
        The fit method only trains the generator component. The target values y
        are not used in this step.
        """
        if y is not None:
            warnings.warn("y won't be used")
        if self.centered:
            X_ = StandardScaler().fit_transform(X)
        else:
            X_ = X
        self.generator.fit(X_)
        return self

    def _check_fit(self):
        try:
            self.generator._check_fit()
        except ValueError as exc:
            raise ValueError(
                "The Model-X Knockoff requires to be fitted before computing importance"
            ) from exc

    def importance(self, X, y):
        """
        Calculate feature importance scores using Model-X knockoffs.

        This method generates knockoff variables and computes test statistics to measure
        feature importance. For multiple repeats, the scores are averaged across repeats
        to improve stability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data matrix where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        importances_ : ndarray of shape (n_features,)
            Feature importance scores for each feature.
            Higher absolute values indicate higher importance.

        Notes
        -----
        The method generates knockoff variables that satisfy the exchangeability property
        and computes test statistics comparing original features against their knockoffs.
        When n_repeat > 1, multiple sets of knockoffs are generated and results are averaged.
        """
        self._check_fit()

        if self.centered:
            X_ = StandardScaler().fit_transform(X)
        else:
            X_ = X

        X_tildes = self.generator.sample(
            n_repeats=self.n_repeat, random_state=self.randoms_state
        )

        parallel = Parallel(self.n_jobs, verbose=self.joblib_verbose)
        self.test_scores_ = np.array(
            parallel(
                delayed(self.statistical_test)(X_, X_tildes[i], y)
                for i in range(self.n_repeat)
            )
        )
        self.test_scores_ = np.array(self.test_scores_)

        self.importances_ = np.mean(self.test_scores_, axis=0)
        self.pvalues_ = np.mean(
            [
                _empirical_knockoff_pval(self.test_scores_[i])
                for i in range(self.n_repeat)
            ],
            axis=0,
        )
        return self.importances_

    def fit_importance(self, X, y, cv=None):
        """
        Fits the model to the data and computes feature importance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data matrix where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            The target values.
        cv : None or cross-validation generator, default=None
            Cross-validation parameter. Not used in this method.
            A warning will be issued if provided.

        Returns
        -------
        importances_ : ndarray of shape (n_features,)
            Feature importance scores (p-values) for each feature.
            Lower values indicate higher importance. Values range from 0 to 1.

        Notes
        -----
        This method combines the fit and importance computation steps.
        It first fits the generator to X and then computes importance scores
        by comparing observed test statistics against permuted ones.

        See Also
        --------
        fit : Method for fitting the generator only
        importance : Method for computing importance scores only
        """
        if cv is not None:
            warnings.warn("cv won't be used")

        self.fit(X)
        return self.importance(X, y)

    def fdr_selection(
        self,
        fdr,
        fdr_control="bhq",
        evalues=False,
        reshaping_function=None,
        adaptive_aggregation=False,
        gamma=0.5,
    ):
        """
        Performs feature selection based on False Discovery Rate (FDR) control.

        This method selects features by controlling the FDR using either p-values or e-values
        derived from test scores. It supports different FDR control methods and optional
        adaptive aggregation of the statistical values.

        Parameters
        ----------
        fdr : float, default=None
            The target false discovery rate level (between 0 and 1)
        fdr_control: string, default="bhq"
            The FDR control method to use. Options are:
            - "bhq": Benjamini-Hochberg procedure
            - 'bhy': Benjamini-Hochberg-Yekutieli procedure
            - "ebh": e-BH procedure (only for e-values)
        evalues: boolean, default=False
            If True, uses e-values for selection. If False, uses p-values.
        reshaping_function: callable, default=None
            Reshaping function for BHY method, default uses sum of reciprocals
        adaptive_aggregation: boolean, default=False
            If True, uses adaptive weights for p-value aggregation.
            Only applicable when evalues=False.
        gamma: boolean, default=0.5
            The gamma parameter for quantile aggregation of p-values.
            Only used when evalues=False.

        Returns
        -------
        numpy.ndarray
            Boolean array indicating selected features (True for selected, False for not selected)

        Raises
        ------
        AssertionError
            If `test_scores_` is None or if incompatible combinations of parameters are provided
        """
        self._check_importance()
        assert (
            self.test_scores_ is not None
        ), "this method doesn't support selection base on FDR"

        if self.test_scores_.shape[0] == 1:
            self.threshold_fdr_ = _knockoff_threshold(self.test_scores_, fdr=fdr)
            selected = self.test_scores_[0] >= self.threshold_fdr_
        elif not evalues:
            assert fdr_control != "ebh", "for p-value, the fdr control can't be 'ebh'"
            pvalues = np.array(
                [
                    _empirical_knockoff_pval(test_score)
                    for test_score in self.test_scores_
                ]
            )
            self.aggregated_pval_ = quantile_aggregation(
                pvalues, gamma=gamma, adaptive=adaptive_aggregation
            )
            self.threshold_fdr_ = fdr_threshold(
                self.aggregated_pval_,
                fdr=fdr,
                method=fdr_control,
                reshaping_function=reshaping_function,
            )
            selected = self.aggregated_pval_ <= self.threshold_fdr_
        else:
            assert fdr_control == "ebh", "for e-value, the fdr control need to be 'ebh'"
            evalues = []
            for test_score in self.test_scores_:
                ko_threshold = _knockoff_threshold(test_score, fdr=fdr)
                evalues.append(_empirical_knockoff_eval(test_score, ko_threshold))
            self.aggregated_eval_ = np.mean(evalues, axis=0)
            self.threshold_fdr_ = fdr_threshold(
                self.aggregated_eval_,
                fdr=fdr,
                method=fdr_control,
                reshaping_function=reshaping_function,
            )
            selected = self.aggregated_eval_ >= self.threshold_fdr_
        return selected


def _knockoff_threshold(test_score, fdr=0.1):
    """
    Calculate the knockoff threshold based on the procedure stated in the article.

    Original code:
    https://github.com/msesia/knockoff-filter/blob/master/R/knockoff/R/knockoff_filter.R

    Parameters
    ----------
    test_score : 1D ndarray, shape (n_features, )
        Vector of test statistic.

    fdr : float
        Desired controlled FDR (false discovery rate) level.

    Returns
    -------
    threshold : float or np.inf
        Threshold level.
    """
    offset = 1  # Offset equals 1 is the knockoff+ procedure.

    threshold_mesh = np.sort(np.abs(test_score[test_score != 0]))
    np.concatenate(
        [[0], threshold_mesh, [np.inf]]
    )  # if there is no solution, the threshold is inf
    # find the right value of t for getting a good fdr
    # Equation 1.8 of barber2015controlling and 3.10 in Candès 2018
    threshold = 0.0
    for threshold in threshold_mesh:
        false_pos = np.sum(test_score <= -threshold)
        selected = np.sum(test_score >= threshold)
        if (offset + false_pos) / np.maximum(selected, 1) <= fdr:
            break
    return threshold


def _empirical_knockoff_pval(test_score):
    """
    Compute the empirical p-values from the knockoff+ test.

    Parameters
    ----------
    test_score : 1D ndarray, shape (n_features, )
        Vector of test statistics.

    Returns
    -------
    pvals : 1D ndarray, shape (n_features, )
        Vector of empirical p-values.
    """
    pvals = []
    n_features = test_score.size

    offset = 1  # Offset equals 1 is the knockoff+ procedure.

    test_score_inv = -test_score
    for i in range(n_features):
        if test_score[i] <= 0:
            pvals.append(1)
        else:
            pvals.append(
                (offset + np.sum(test_score_inv >= test_score[i])) / n_features
            )

    return np.array(pvals)


def _empirical_knockoff_eval(test_score, ko_threshold):
    """
    Compute the empirical e-values from the knockoff test.

    Parameters
    ----------
    test_score : 1D ndarray, shape (n_features, )
        Vector of test statistics.

    ko_threshold : float
        Threshold level.

    Returns
    -------
    evals : 1D ndarray, shape (n_features, )
        Vector of empirical e-values.
    """
    evals = []
    n_features = test_score.size

    offset = 1  # Offset equals 1 is the knockoff+ procedure.

    for i in range(n_features):
        if test_score[i] < ko_threshold:
            evals.append(0)
        else:
            evals.append(n_features / (offset + np.sum(test_score <= -ko_threshold)))

    return np.array(evals)


def model_x_knockoff(
    X,
    y,
    generator=GaussianKnockoffs(),
    statistical_test=lasso_statistic_with_sampling,
    n_repeat=1,
    centered=True,
    random_state=None,
    joblib_verbose=0,
    memory=None,
    n_jobs=1,
    fdr=0.1,
    fdr_control="bhq",
    evalues=False,
    reshaping_function=None,
    adaptive_aggregation=False,
    gamma=0.5,
):
    methods = ModelXKnockoff(
        ko_generator=generator,
        statistical_test=statistical_test,
        n_repeat=n_repeat,
        centered=centered,
        random_state=random_state,
        joblib_verbose=joblib_verbose,
        memory=memory,
        n_jobs=n_jobs,
    )
    methods.fit_importance(X, y)
    selected = methods.fdr_selection(
        fdr=fdr,
        fdr_control=fdr_control,
        evalues=evalues,
        reshaping_function=reshaping_function,
        adaptive_aggregation=adaptive_aggregation,
        gamma=gamma,
    )
    return selected, methods.importances_, methods.pvalues_


# use the docstring of the class for the function
model_x_knockoff.__doc__ = _aggregate_docstring(
    [
        ModelXKnockoff.__doc__,
        ModelXKnockoff.__init__.__doc__,
        ModelXKnockoff.fit_importance.__doc__,
        ModelXKnockoff.fdr_selection.__doc__,
    ],
    """
    Returns
    -------
    selection: binary array-like of shape (n_features)
        Binary array of the selected features
    importance : array-like of shape (n_features)
        The computed feature importance scores.
    pvalues : array-like of shape (n_features)
        The computed significant of feature for the prediction.
    """,
)
