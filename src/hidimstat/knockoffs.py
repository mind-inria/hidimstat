import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_memory

from hidimstat._utils.docstring import _aggregate_docstring
from hidimstat._utils.utils import check_random_state, seed_estimator
from hidimstat.base_variable_importance import BaseVariableImportance
from hidimstat.statistical_tools.aggregation import quantile_aggregation
from hidimstat.statistical_tools.gaussian_knockoffs import GaussianKnockoffs
from hidimstat.statistical_tools.multiple_testing import fdr_threshold


def set_alpha_max_lasso_path(estimator, X, X_tilde, y, n_alphas=20):
    """
    Configure a LassoCV estimator's regularization path for concatenated features.

    Sets estimator.alphas based on the combined design matrix formed by
    concatenating X and X_tilde column-wise. The maximum alpha is computed as

        alpha_max = max(X_ko.T @ y) / (2 * n_features)

    and an alpha grid of length n_alphas is created between
    alpha_max * exp(-n_alphas) and alpha_max.

    Parameters
    ----------
    estimator : sklearn.linear_model.LassoCV
        LassoCV instance to configure. This function modifies estimator.alphas
        in-place and returns the same estimator.
    X : array-like, shape (n_samples, n_features)
        Original feature matrix.
    X_tilde : array-like, shape (n_samples, n_features)
        Knockoff/auxiliary feature matrix (must have same n_features as X).
    y : array-like, shape (n_samples,)
        Target vector.
    n_alphas : int, default=20
        Number of alpha values to generate for the Lasso path.

    Returns
    -------
    estimator : sklearn.linear_model.LassoCV
        The same estimator with estimator.alphas set.

    Raises
    ------
    TypeError
        If estimator is not an instance/name-matching LassoCV.

    Notes
    -----
    - The function expects X and X_tilde already preprocessed (e.g. centered/scaled)
      as appropriate for the estimator.
    - The generated alpha grid is deterministic given X, X_tilde and y.
    """
    if type(estimator).__name__ != "LassoCV":
        raise TypeError("You should not use this function to configure the estimator")

    n_features = X.shape[1]
    X_ko = np.column_stack([X, X_tilde])
    alpha_max = np.max(np.dot(X_ko.T, y)) / (2 * n_features)
    alphas = np.linspace(alpha_max * np.exp(-n_alphas), alpha_max, n_alphas)
    estimator.alphas = alphas
    return estimator


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
    estimator : estimator, default=LassoCV(...)
        Estimator used to compute knockoff statistics. Must expose coefficients via
        `coef_` (or `best_estimator_.coef_` for CV wrappers) after fit.
    ko_generator : object
        Knockoff generator implementing fit(X) and sample(n_repeats, random_state).
    n_repeats: int, default=1
        Number of knockoff draws to average over.
    centered : bool, default=True
        If True, standardize X before fitting the generator and computing statistics.
    preconfigure_lasso_path : bool, default=True
        An optional function is called to configure the LassoCV estimator's regularization path.
        The maximum alpha is computed as `alpha_max = max(X_ko.T @ y) / (2 * n_features)`
        and an alpha grid of length n_alphas is created between
        alpha_max * exp(-n_alphas) and alpha_max.
    random_state : int or None, default=None
        Random seed forwarded to the knockoff generator sampling.
    joblib_verbose : int, default=0
        Verbosity level for parallel jobs.
    memory : str, joblib.Memory or None, default=None
        Caching backend for expensive operations.
    n_jobs : int, default=1
        Number of parallel jobs (automatically capped to n_repeats).

    Attributes
    ----------
    importances_ : ndarray, shape (n_repeats, n_features)
        Test statistics for each repeat.
    pvalues_ : ndarray, shape (n_repeats, n_features)
        Empirical p-values for each repeat.
    threshold_fdr_ : float
        Threshold computed by the FDR selection procedure.
    aggregated_pval_ : ndarray or None
        Aggregated p-values (when using p-value aggregation).
    aggregated_eval_ : ndarray or None
        Aggregated e-values (when using e-value aggregation).
    estimators_ : list of estimators
        List of fitted estimators on the concatenated design matrices for each repeat.
    n_features_ : int
        Number of features on which the model was fitted.

    Notes
    -----
    Use the model_x_knockoff function for a functional interface that wraps this
    class. The class focuses on generator fitting, repeated knockoff sampling,
    computing statistics and performing FDR-based selection.
    """

    def __init__(
        self,
        estimator=LassoCV(
            max_iter=200000,
            n_jobs=1,
            verbose=0,
            cv=KFold(n_splits=5, shuffle=True, random_state=0),
            random_state=1,
            tol=1e-6,
        ),
        ko_generator=GaussianKnockoffs(),
        n_repeats=1,
        centered=True,
        preconfigure_lasso_path=True,
        random_state=None,
        joblib_verbose=0,
        memory=None,
        n_jobs=1,
    ):
        super().__init__()
        self.generator = ko_generator
        assert n_repeats > 0, "n_samplings must be positive"
        self.n_repeats = n_repeats
        self.centered = centered
        # parameter for statistical test base on linear model
        self.estimator = estimator
        self.preconfigure_lasso_path = preconfigure_lasso_path

        self.randoms_state = random_state
        self.memory = check_memory(memory)
        self.joblib_verbose = joblib_verbose
        # unnecessary to have n_jobs > number of bootstraps
        self.n_jobs = min(n_repeats, n_jobs)

        self.importances_ = None
        self.threshold_fdr_ = None
        self.aggregated_eval_ = None
        self.aggregated_pval_ = None
        self.estimators_ = None
        self.n_features_ = None

    def fit(self, X, y):
        """
        Fit the knockoff generator and estimators to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data matrix where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        rng = check_random_state(self.randoms_state)
        if self.centered:
            X_ = StandardScaler().fit_transform(X)
        else:
            X_ = X

        self.generator.fit(X_)
        X_tildes = self.generator.sample(
            n_repeats=self.n_repeats, random_state=self.randoms_state
        )

        self.estimators_ = Parallel(self.n_jobs, verbose=self.joblib_verbose)(
            delayed(self._joblib_fit_estimator)(
                self.estimator,
                X_,
                X_tildes[i],
                y,
                spawned_rng,
                preconfigure_lasso_path=self.preconfigure_lasso_path,
            )
            for i, spawned_rng in enumerate(rng.spawn(self.n_repeats))
        )
        self.n_features_ = X.shape[1]
        return self

    def _check_fit(self):
        if self.estimators_ is None:
            raise ValueError(
                "The Model-X Knockoff requires to be fitted before computing importance"
            )

    def importance(self, X=None, y=None):
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
        When n_repeats > 1, multiple sets of knockoffs are generated and results are averaged.
        """
        if X is not None:
            warnings.warn("X won't be used")
        if y is not None:
            warnings.warn("y won't be used")
        self._check_fit()

        self.importances_ = self.lasso_coefficient_difference_statistic(
            self.estimators_, self.n_features_
        )
        self.pvalues_ = np.array(
            [
                self._empirical_knockoff_pval(self.importances_[i])
                for i in range(self.n_repeats)
            ]
        )
        return self.importances_

    def fit_importance(self, X, y):
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
        self.fit(X, y)
        return self.importance()

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
            If `importances_` is None or if incompatible combinations of parameters are provided
        """
        self._check_importance()
        assert (
            self.importances_ is not None
        ), "this method doesn't support selection base on FDR"

        if self.importances_.shape[0] == 1:
            self.threshold_fdr_ = self.knockoff_threshold(self.importances_, fdr=fdr)
            selected = self.importances_[0] >= self.threshold_fdr_
        elif not evalues:
            assert fdr_control != "ebh", "for p-value, the fdr control can't be 'ebh'"
            pvalues = np.array(
                [
                    self._empirical_knockoff_pval(test_score)
                    for test_score in self.importances_
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
            for test_score in self.importances_:
                ko_threshold = self.knockoff_threshold(test_score, fdr=fdr)
                evalues.append(self._empirical_knockoff_eval(test_score, ko_threshold))
            self.aggregated_eval_ = np.mean(evalues, axis=0)
            self.threshold_fdr_ = fdr_threshold(
                self.aggregated_eval_,
                fdr=fdr,
                method=fdr_control,
                reshaping_function=reshaping_function,
            )
            selected = self.aggregated_eval_ >= self.threshold_fdr_
        return selected

    @staticmethod
    def _joblib_fit_estimator(
        estimator, X, X_tilde, y, random_state, preconfigure_lasso_path=False
    ):
        """
        Single fit of the estimator on the concatenated design matrix [X, X_tilde].
        """
        estimator_ = clone(estimator)
        # Preconfigure the estimator if needed
        if preconfigure_lasso_path:
            if hasattr(estimator_, "alphas") and (estimator_.alphas is not None):
                n_alphas = len(estimator_.alphas)
            elif hasattr(estimator_, "n_alphas") and (estimator_.n_alphas is not None):
                n_alphas = estimator_.n_alphas
            else:
                n_alphas = 10
            estimator_ = set_alpha_max_lasso_path(
                estimator_, X, X_tilde, y, n_alphas=n_alphas
            )

        X_ = np.column_stack([X, X_tilde])
        estimator_ = seed_estimator(estimator_, random_state)
        estimator_.fit(X_, y)
        return estimator_

    @staticmethod
    def lasso_coefficient_difference_statistic(estimators, n_features):
        """
        Compute the Lasso Coefficient-Difference (LCD) statistic from a fitted estimator.
        Given a list of fitted estimators on the concatenated design matrix [X, X_tilde],
        this function computes the knockoff statistic for each original feature across
        repeats:

        .. math::
            W_j = |\\beta_j| - |\\beta_j'|

        where :math:`\\beta_j` and :math:`\\beta_j'` are the fitted coefficients for the original feature j
        and its knockoff counterpart j'.

        Parameters
        ----------
        estimators : list of estimators
            List of fitted estimators on the concatenated design matrix [X, X_tilde].
            Each estimator must expose coefficients via `coef_` or
            `best_estimator_.coef_`.
        n_features : int
            Number of original features (not including knockoffs).

        Returns
        -------
        test_statistic : ndarray, shape (n_repeats, n_features)
            Knockoff statistics :math:`W_j` for each original feature across repeats. The number
            of repeats corresponds to the length of the estimators list.
        """
        test_statistic_list = []
        for estimator in estimators:
            if hasattr(estimator, "coef_"):
                coef = np.ravel(estimator.coef_)
            elif hasattr(estimator, "best_estimator_") and hasattr(
                estimator.best_estimator_, "coef_"
            ):
                coef = np.ravel(estimator.best_estimator_.coef_)  # for CV object
            else:
                raise TypeError("estimator should be linear")
            statistic_tmp = np.abs(coef[:n_features]) - np.abs(coef[n_features:])
            test_statistic_list.append(statistic_tmp)

        test_statistic = np.array(test_statistic_list)
        return test_statistic

    @staticmethod
    def knockoff_threshold(test_score, fdr=0.1):
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

    @staticmethod
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

    @staticmethod
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
                evals.append(
                    n_features / (offset + np.sum(test_score <= -ko_threshold))
                )

        return np.array(evals)


def model_x_knockoff(
    X,
    y,
    estimator=LassoCV(max_iter=200000),
    generator=GaussianKnockoffs(),
    n_repeats=1,
    centered=True,
    random_state=None,
    preconfigure_lasso_path=True,
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
        n_repeats=n_repeats,
        centered=centered,
        estimator=estimator,
        preconfigure_lasso_path=preconfigure_lasso_path,
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
