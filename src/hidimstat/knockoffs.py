import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_memory

from hidimstat._utils.utils import check_random_state
from hidimstat.gaussian_knockoff import (
    gaussian_knockoff_generation,
    repeat_gaussian_knockoff_generation,
)
from hidimstat.statistical_tools.aggregation import quantile_aggregation
from hidimstat.statistical_tools.multiple_testing import fdr_threshold


def preconfigure_estimator_LassoCV(estimator, X, X_tilde, y, n_alphas=20):
    """
    Configure the estimator for Model-X knockoffs.

    This function sets up the regularization path for the Lasso estimator
    based on the input data and the number of alphas to use. The regularization
    path is defined by a sequence of alpha values, which control the amount
    of shrinkage applied to the coefficient estimates.

    Parameters
    ----------
    estimator : sklearn.linear_model.LassoCV
        The Lasso estimator to configure.

    X : 2D ndarray (n_samples, n_features)
        The original design matrix.

    X_tilde : 2D ndarray (n_samples, n_features)
        The knockoff design matrix.

    y : 1D ndarray (n_samples, )
        The target vector.

    n_alphas : int, default=10
        The number of alpha values to use to instantiate the cross-validation.

    Returns
    -------
    estimator : sklearn.linear_model.LassoCV
        The configured estimator.

    Raises
    ------
    TypeError
        If estimator is not an instance of LassoCV.

    Notes
    -----
    The alpha values are calculated based on the combined design matrix [X, X_tilde].
    alpha_max is set to max(X_ko.T @ y)/(2*n_features).
    """
    if type(estimator).__name__ != "LassoCV":
        raise TypeError("You should not use this function to configure the estimator")

    n_features = X.shape[1]
    X_ko = np.column_stack([X, X_tilde])
    alpha_max = np.max(np.dot(X_ko.T, y)) / (2 * n_features)
    alphas = np.linspace(alpha_max * np.exp(-n_alphas), alpha_max, n_alphas)
    estimator.alphas = alphas
    return estimator


def model_x_knockoff(
    X,
    y,
    estimator=LassoCV(
        n_jobs=None,
        verbose=0,
        max_iter=200000,
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        tol=1e-6,
    ),
    preconfigure_estimator=preconfigure_estimator_LassoCV,
    fdr=0.1,
    centered=True,
    cov_estimator=LedoitWolf(assume_centered=True),
    joblib_verbose=0,
    n_bootstraps=1,
    n_jobs=1,
    random_state=None,
    tol_gauss=1e-14,
    memory=None,
):
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
    assert n_bootstraps > 0, "the number of bootstraps should at least higher than 1"
    memory = check_memory(memory)
    # unnecessary to have n_jobs > number of bootstraps
    n_jobs = min(n_bootstraps, n_jobs)
    parallel = Parallel(n_jobs, verbose=joblib_verbose)

    # get the seed for the different run
    rng = check_random_state(random_state)
    children_rng = rng.spawn(n_bootstraps)

    if centered:
        X = StandardScaler().fit_transform(X)

    # estimation of X distribution
    # original implementation:
    # https://github.com/msesia/knockoff-filter/blob/master/R/knockoff/R/create_second_order.R
    mu = X.mean(axis=0)
    sigma = cov_estimator.fit(X).covariance_

    # Create knockoff variables
    X_tilde, mu_tilde, sigma_tilde_decompose = memory.cache(
        gaussian_knockoff_generation
    )(X, mu, sigma, seed=children_rng[0], tol=tol_gauss)

    if n_bootstraps == 1:
        X_tildes = [X_tilde]
    else:
        X_tildes = parallel(
            delayed(repeat_gaussian_knockoff_generation)(
                mu_tilde,
                sigma_tilde_decompose,
                seed=seed,
            )
            for seed in children_rng[1:]
        )
        X_tildes.insert(0, X_tilde)

    results = parallel(
        delayed(memory.cache(_stat_coefficient_diff))(
            X, X_tildes[i], y, clone(estimator), fdr, preconfigure_estimator
        )
        for i in range(n_bootstraps)
    )
    test_scores, threshold, selected = zip(*results)

    if n_bootstraps == 1:
        return selected[0], test_scores[0], threshold[0], X_tildes[0]
    else:
        return selected, test_scores, threshold, X_tildes


def model_x_knockoff_pvalue(test_score, fdr=0.1, fdr_control="bhq"):
    """
    This function implements the computation of the empirical p-values

    Parameters
    ----------
    test_score : 1D array, (n_features, )
        A vector of test statistics.

    fdr : float, default=0.1
        The desired controlled False Discovery Rate (FDR) level.

    fdr_control : str, default="bhq"
        The method used to control the False Discovery Rate.
        Available methods are:
        * 'bhq': Standard Benjamini-Hochberg :footcite:`benjamini1995controlling,bhy_2001`
        * 'bhy': Benjamini-Hochberg-Yekutieli :footcite:p:`bhy_2001`
        * 'ebh': e-Benjamini-Hochberg :footcite:`wang2022false`

    Returns
    -------
    selected : 1D array, int
        A vector of indices of the selected variables.

    pvals : 1D array, (n_features, )
        A vector of empirical p-values.

    Notes
    -----
    This function calculates the empirical p-values based on the test statistics and the
    desired FDR level. It then identifies the selected variables based on the p-values.
    """
    pvals = _empirical_knockoff_pval(test_score)
    threshold = fdr_threshold(pvals, fdr=fdr, method=fdr_control)
    selected = np.where(pvals <= threshold)[0]
    return selected, pvals


def model_x_knockoff_bootstrap_e_value(test_scores, ko_threshold, fdr=0.1):
    """
    This function implements the computation of the empirical e-values
    from knockoff test and aggregates them using the e-BH procedure.

    Parameters
    ----------
    test_scores : 2D array, (n_bootstraps, n_features)
        A matrix of test statistics for each bootstrap sample.

    ko_threshold : float
        Threshold level.

    fdr : float, default=0.1
        The desired controlled False Discovery Rate (FDR) level.

    Returns
    -------
    selected : 1D array, int
        A vector of indices of the selected variables.

    aggregated_eval : 1D array, (n_features, )
        A vector of aggregated empirical e-values.

    evals : 2D array, (n_bootstraps, n_features)
        A matrix of empirical e-values for each bootstrap sample.

    Notes
    -----
    This function calculates the empirical e-values based on the test statistics and the
    desired FDR level. It then aggregates the e-values using the e-BH procedure and identifies
    the selected variables based on the aggregated e-values.
    """
    n_bootstraps = len(test_scores)
    evals = np.array(
        [
            _empirical_knockoff_eval(test_scores[i], ko_threshold[i])
            for i in range(n_bootstraps)
        ]
    )

    aggregated_eval = np.mean(evals, axis=0)
    threshold = fdr_threshold(aggregated_eval, fdr=fdr, method="ebh")
    selected = np.where(aggregated_eval >= threshold)[0]

    return selected, aggregated_eval, evals


def model_x_knockoff_bootstrap_quantile(
    test_scores,
    fdr=0.1,
    fdr_control="bhq",
    reshaping_function=None,
    adaptive_aggregation=False,
    gamma=0.5,
):
    """
    This function implements the computation of the empirical p-values
    from knockoff test and aggregates them using the quantile aggregation procedure.

    Parameters
    ----------
    test_scores : 2D array, (n_bootstraps, n_features)
        A matrix of test statistics for each bootstrap sample.

    fdr : float, default=0.1
        The desired controlled False Discovery Rate (FDR) level.

    fdr_control : str, default="bhq"
        The method used to control the False Discovery Rate.
        Available methods are:
        * 'bhq': Standard Benjamini-Hochberg :footcite:`benjamini1995controlling,bhy_2001`
        * 'bhy': Benjamini-Hochberg-Yekutieli :footcite:p:`bhy_2001`
        * 'ebh': e-Benjamini-Hochberg :footcite:`wang2022false`

    reshaping_function : function or None, default=None
        A function used to reshape the aggregated p-values before controlling the FDR.

    adaptive_aggregation : bool, default=False
        Whether to use adaptive quantile aggregation.

    gamma : float, default=0.5
        The quantile level (between 0 and 1) used for aggregation.
        For non-adaptive aggregation, a single gamma value is used.
        For adaptive aggregation, this is the starting point for the grid search
        over gamma values.

    Returns
    -------
    selected : 1D array, int
        A vector of indices of the selected variables.

    aggregated_pval : 1D array, (n_features, )
        A vector of aggregated empirical p-values.

    pvals : 2D array, (n_bootstraps, n_features)
        A matrix of empirical p-values for each bootstrap sample.

    Notes
    -----
    This function calculates the empirical p-values based on the test statistics and the
    desired FDR level. It then aggregates the p-values using the quantile aggregation
    procedure and identifies the selected variables based on the aggregated p-values.
    """
    n_bootstraps = len(test_scores)
    pvals = np.array(
        [_empirical_knockoff_pval(test_scores[i]) for i in range(n_bootstraps)]
    )

    aggregated_pval = quantile_aggregation(
        pvals, gamma=gamma, adaptive=adaptive_aggregation
    )

    threshold = fdr_threshold(
        aggregated_pval,
        fdr=fdr,
        method=fdr_control,
        reshaping_function=reshaping_function,
    )
    selected = np.where(aggregated_pval <= threshold)[0]

    return selected, aggregated_pval, pvals


def _stat_coefficient_diff(X, X_tilde, y, estimator, fdr, preconfigure_estimator=None):
    """
    Compute the Lasso Coefficient-Difference (LCD) statistic by comparing original and knockoff coefficients.

    This function fits a model on the concatenated original and knockoff features, then
    calculates test statistics based on the difference between coefficient magnitudes.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Original feature matrix.

    X_tilde : ndarray of shape (n_samples, n_features)
        Knockoff feature matrix.

    y : ndarray of shape (n_samples,)
        Target values.

    estimator : estimator object
        Scikit-learn estimator with fit() method and coef_ attribute.
        Common choices include LassoCV, LogisticRegressionCV.

    fdr : float
        Target false discovery rate level between 0 and 1.

    preconfigure_estimator : callable, default=None
        Optional function to configure estimator parameters before fitting.
        Called with arguments (estimator, X, X_tilde, y).

    Returns
    -------
    test_score : ndarray of shape (n_features,)
        Feature importance scores computed as |beta_j| - |beta_j'|
        where beta_j and beta_j' are original and knockoff coefficients.

    ko_thr : float
        Knockoff threshold value used for feature selection.

    selected : ndarray
        Indices of features with test_score >= ko_thr.

    Notes
    -----
    The test statistic follows Equation 1.7 in Barber & Candès (2015) and
    Equation 3.6 in Candès et al. (2018).
    """
    n_samples, n_features = X.shape
    X_ko = np.column_stack([X, X_tilde])
    if preconfigure_estimator is not None:
        estimator = preconfigure_estimator(estimator, X, X_tilde, y)
    estimator.fit(X_ko, y)
    if hasattr(estimator, "coef_"):
        coef = np.ravel(estimator.coef_)
    elif hasattr(estimator, "best_estimator_") and hasattr(
        estimator.best_estimator_, "coef_"
    ):
        coef = np.ravel(estimator.best_estimator_.coef_)  # for CV object
    else:
        raise TypeError("estimator should be linear")
    # Equation 1.7 in barber2015controlling or 3.6 of candes2018panning
    test_score = np.abs(coef[:n_features]) - np.abs(coef[n_features:])

    # Compute the threshold level and select the important variables
    ko_thr = _knockoff_threshold(test_score, fdr=fdr)
    selected = np.where(test_score >= ko_thr)[0]

    return test_score, ko_thr, selected


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
