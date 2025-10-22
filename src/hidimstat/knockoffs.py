import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_memory

from hidimstat._utils.docstring import _aggregate_docstring
from hidimstat.base_variable_importance import BaseVariableImportance
from hidimstat.statistical_tools.aggregation import quantile_aggregation
from hidimstat.statistical_tools.gaussian_knockoffs import GaussianKnockoffs
from hidimstat.statistical_tools.multiple_testing import fdr_threshold


def preconfigure_lasso_path(estimator, X, X_tilde, y, n_alphas=20):
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
    ko_generator : object
        Knockoff generator implementing fit(X) and sample(n_repeats, random_state).
    n_repeat : int, default=1
        Number of knockoff draws to average over.
    centered : bool, default=True
        If True, standardize X before fitting the generator and computing statistics.
    test_linear_model : estimator, default=LassoCV(...)
        Estimator used to compute knockoff statistics. Must expose coefficients via
        `coef_` (or `best_estimator_.coef_` for CV wrappers) after fit.
    test_preconfigure_model : callable or None, default=preconfigure_lasso_path
        An optional function is called to configure the LassoCV estimator's regularization path. 
        The maximum alpha is computed as
            alpha_max = max(X_ko.T @ y) / (2 * n_features)
        and an alpha grid of length n_alphas is created between
        alpha_max * exp(-n_alphas) and alpha_max.
    random_state : int or None, default=None
        Random seed forwarded to the knockoff generator sampling.
    joblib_verbose : int, default=0
        Verbosity level for parallel jobs.
    memory : str, joblib.Memory or None, default=None
        Caching backend for expensive operations.
    n_jobs : int, default=1
        Number of parallel jobs (automatically capped to n_repeat).

    Attributes
    ----------
    importances_ : ndarray, shape (n_features,)
        Averaged test statistics across repeats.
    pvalues_ : ndarray, shape (n_features,)
        Averaged empirical p-values across repeats.
    test_scores_ : ndarray, shape (n_repeat, n_features)
        Raw test scores for each repeat.
    threshold_fdr_ : float
        Threshold computed by the FDR selection procedure.
    aggregated_pval_ : ndarray or None
        Aggregated p-values (when using p-value aggregation).
    aggregated_eval_ : ndarray or None
        Aggregated e-values (when using e-value aggregation).

    Notes
    -----
    Use the model_x_knockoff function for a functional interface that wraps this
    class. The class focuses on generator fitting, repeated knockoff sampling,
    computing statistics and performing FDR-based selection.
    """

    def __init__(
        self,
        ko_generator=GaussianKnockoffs(),
        n_repeat=1,
        centered=True,
        test_linear_model=LassoCV(
            n_jobs=1,
            verbose=0,
            max_iter=200000,
            cv=KFold(n_splits=5, shuffle=True, random_state=0),
            random_state=1,
            tol=1e-6,
        ),
        test_preconfigure_model=preconfigure_lasso_path,
        random_state=None,
        joblib_verbose=0,
        memory=None,
        n_jobs=1,
    ):
        super().__init__()
        self.generator = ko_generator
        assert n_repeat > 0, "n_samplings must be positive"
        self.n_repeat = n_repeat
        self.centered = centered
        # parameter for statistical test base on linear model
        self.test_linear_model = test_linear_model
        self.test_preconfigure_model = test_preconfigure_model

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
        self.test_scores_ = parallel(
            delayed(job_lib_lasso_statistic)(
                X_,
                X_tildes[i],
                y,
                clone(self.test_linear_model),
                self.test_preconfigure_model,
            )
            for i in range(self.n_repeat)
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


def job_lib_lasso_statistic(X, X_tilde, y, test_linear_model, test_preconfgure_model):
    """
    Compute the Lasso Coefficient-Difference (LCD) statistic.

    Fits the provided linear estimator on the concatenated design matrix [X, X_tilde]
    and returns the knockoff statistic for each original feature:

        W_j = |beta_j| - |beta_j'|

    where beta_j and beta_j' are the fitted coefficients for the original feature j
    and its knockoff counterpart j'.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Original feature matrix.
    X_tilde : ndarray, shape (n_samples, n_features)
        Knockoff feature matrix.
    y : ndarray, shape (n_samples,)
        Target vector.
    test_linear_model : estimator instance
        Estimator to fit on the concatenated matrix. Must implement fit(X, y) and expose
        coefficients via `coef_` after fitting, or via `best_estimator_.coef_` for CV wrappers.
    test_preconfgure_model : callable or None
        If provided, callable(test_linear_model, X, X_tilde, y) should return a configured
        estimator (or modify it in-place). If None, `test_linear_model` is used as-is.

    Returns
    -------
    test_score : ndarray, shape (n_features,)
        Knockoff statistics W_j for each original feature. Larger values indicate stronger
        evidence that the original feature is more important than its knockoff.

    Raises
    ------
    TypeError
        If the fitted estimator does not provide coefficients via `coef_` or
        `best_estimator_.coef_`.

    Notes
    -----
    The function stacks X and X_tilde column-wise before fitting. Coefficients are flattened
    with `np.ravel` and the statistic follows the standard knockoff LCD definition.
    """
    n_samples, n_features = X.shape
    X_ko = np.column_stack([X, X_tilde])
    if test_preconfgure_model is not None:
        linear_model = test_preconfgure_model(test_linear_model, X, X_tilde, y)
    else:
        linear_model = test_linear_model
    linear_model.fit(X_ko, y)
    if hasattr(linear_model, "coef_"):
        coef = np.ravel(linear_model.coef_)
    elif hasattr(linear_model, "best_estimator_") and hasattr(
        linear_model.best_estimator_, "coef_"
    ):
        coef = np.ravel(linear_model.best_estimator_.coef_)  # for CV object
    else:
        raise TypeError("estimator should be linear")
    # Equation 1.7 in barber2015controlling or 3.6 of candes2018panning
    test_score = np.abs(coef[:n_features]) - np.abs(coef[n_features:])

    return test_score


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
    n_repeat=1,
    centered=True,
    random_state=None,
    test_linear_model=LassoCV(
        n_jobs=1,
        verbose=0,
        max_iter=200000,
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        random_state=1,
        tol=1e-6,
    ),
    test_preconfigure_model=preconfigure_lasso_path,
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
        n_repeat=n_repeat,
        centered=centered,
        test_linear_model=test_linear_model,
        test_preconfigure_model=test_preconfigure_model,
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
