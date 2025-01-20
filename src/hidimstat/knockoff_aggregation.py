# -*- coding: utf-8 -*-
# Authors: Binh Nguyen <tuan-binh.nguyen@inria.fr>
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_memory

from .gaussian_knockoff import _estimate_distribution, gaussian_knockoff_generation
from .stat_coef_diff import stat_coef_diff, _coef_diff_threshold
from .utils import fdr_threshold, quantile_aggregation


def knockoff_aggregation(
    X,
    y,
    centered=True,
    shrink=False,
    construct_method="equi",
    fdr=0.1,
    fdr_control="bhq",
    reshaping_function=None,
    offset=1,
    method="quantile",
    statistic="lasso_cv",
    cov_estimator="ledoit_wolf",
    joblib_verbose=0,
    n_bootstraps=25,
    n_jobs=1,
    adaptive_aggregation=False,
    gamma=0.5,
    n_grid_gamma=20,
    verbose=False,
    memory=None,
    random_state=None,
):
    """
    Aggregation of Multiple knockoffs

    This function implements the aggregation of multiple knockoffs introduced by
    :footcite:t:`pmlr-v119-nguyen20a`

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,),
        The target values (class labels in classification, real numbers in
        regression).
    centered : bool, default=True
        Whether to standardize the data before doing the inference procedure.
    shrink : bool, default=False
        Whether to shrink the empirical covariance matrix.
    construct_method : str, default="equi"
        The knockoff construction methods. The options include:
        - "equi" for equi-correlated knockoff
        - "sdp" for optimization scheme
    fdr : float, default=0.1
        The desired controlled FDR level
    fdr_control : srt, default="bhq"
        The control method for False Discovery Rate (FDR). The options include:
        - "bhq" for Standard Benjamini-Hochberg procedure
        - "bhy" for Benjamini-Hochberg-Yekutieli procedure
        - "ebh" for e-BH procedure
    reshaping_function : <class 'function'>, default=None
        The reshaping function defined in :footcite:t:`bhy_2001`.
    offset : int, 0 or 1, optional
        The offset to calculate knockoff threshold, offset = 1 is equivalent to
        knockoff+.
    method : srt, default="quantile"
        The method to compute the statistical measures. The options include:
        - "quantile" for p-values
        - "e-values" for e-values
    statistic : srt, default="lasso_cv"
        The method to calculate knockoff test score.
    cov_estimator : srt, default="ledoitwolf"
        The method of empirical covariance matrix estimation.
    joblib_versobe : int, default=0
       The verbosity level of joblib: if non zero, progress messages are
       printed. Above 50, the output is sent to stdout. The frequency of the
       messages increases with the verbosity level. If it more than 10, all
       iterations are reported.
    n_bootstraps : int, default=25
        The number of bootstrapping iterations.
    n_jobs : int, default=1
        The number of workers for parallel processing.
    adaptive_aggregation : bool, default=False
        Whether to apply the adaptive version of the quantile aggregation method
        as in :footcite:t:`Meinshausen_2008`.
    gamma : float, default=0.5
        The percentile value used for aggregation.
    n_grid_gamma : int, default=20
        Number of gamma grid points for adaptive aggregation.
    verbose : bool, default=False
        Whether to return the corresponding p-values of the variables along with
        the list of selected variables.
    memory : str or joblib.Memory object, default=None
        Used to cache the output of the computation of the clustering
        and the inference. By default, no caching is done. If a string is
        given, it is the path to the caching directory.
    random_state : int, default=None
        Fixing the seeds of the random generator.

    Returns
    -------
    selected : 1D array, int
        The vector of index of selected variables.
    aggregated_pval: 1D array, float
        The vector of aggregated p-values.
    pvals: 1D array, float
        The vector of the corresponding p-values.
    aggregated_eval: 1D array, float
        The vector of aggregated e-values.
    evals: 1D array, float
        The vector of the corresponding e-values.

    References
    ----------
    .. footbibliography::

    """
    # unnecessary to have n_jobs > number of bootstraps
    n_jobs = min(n_bootstraps, n_jobs)

    if centered:
        X = StandardScaler().fit_transform(X)

    mu, Sigma = _estimate_distribution(X, shrink=shrink, cov_estimator=cov_estimator)

    mem = check_memory(memory)
    stat_coef_diff_cached = mem.cache(
        stat_coef_diff, ignore=["n_jobs", "joblib_verbose"]
    )

    if n_bootstraps == 1:
        X_tilde = gaussian_knockoff_generation(
            X, mu, Sigma, method=construct_method, memory=memory, seed=random_state
        )
        ko_stat = stat_coef_diff_cached(X, X_tilde, y, method=statistic)
        pvals = _empirical_pval(ko_stat, offset)
        threshold = fdr_threshold(pvals, fdr=fdr, method=fdr_control)
        selected = np.where(pvals <= threshold)[0]

        if verbose:
            return selected, pvals

        return selected

    if isinstance(random_state, (int, np.int32, np.int64)):
        rng = check_random_state(random_state)
    elif random_state is None:
        rng = check_random_state(0)
    else:
        raise TypeError("Wrong type for random_state")

    seed_list = rng.randint(1, np.iinfo(np.int32).max, n_bootstraps)
    parallel = Parallel(n_jobs, verbose=joblib_verbose)
    X_tildes = parallel(
        delayed(gaussian_knockoff_generation)(
            X, mu, Sigma, method=construct_method, memory=memory, seed=seed
        )
        for seed in seed_list
    )

    ko_stats = parallel(
        delayed(stat_coef_diff_cached)(X, X_tildes[i], y, method=statistic)
        for i in range(n_bootstraps)
    )

    if method == "e-values":
        evals = np.array(
            [_empirical_eval(ko_stats[i], fdr / 2, offset) for i in range(n_bootstraps)]
        )

        aggregated_eval = np.mean(evals, axis=0)
        threshold = fdr_threshold(aggregated_eval, fdr=fdr, method="ebh")
        selected = np.where(aggregated_eval >= threshold)[0]

        if verbose:
            return selected, aggregated_eval, evals

        return selected

    if method == "quantile":
        pvals = np.array(
            [_empirical_pval(ko_stats[i], offset) for i in range(n_bootstraps)]
        )

        aggregated_pval = quantile_aggregation(
            pvals, gamma=gamma, n_grid=n_grid_gamma, adaptive=adaptive_aggregation
        )

        threshold = fdr_threshold(
            aggregated_pval,
            fdr=fdr,
            method=fdr_control,
            reshaping_function=reshaping_function,
        )
        selected = np.where(aggregated_pval <= threshold)[0]

        if verbose:
            return selected, aggregated_pval, pvals

        return selected


def _empirical_pval(test_score, offset=1):
    """
    This function implements the computation of the empirical p-values
    """
    pvals = []
    n_features = test_score.size

    if offset not in (0, 1):
        raise ValueError("'offset' must be either 0 or 1")

    test_score_inv = -test_score
    for i in range(n_features):
        if test_score[i] <= 0:
            pvals.append(1)
        else:
            pvals.append(
                (offset + np.sum(test_score_inv >= test_score[i])) / n_features
            )

    return np.array(pvals)


def _empirical_eval(test_score, fdr=0.1, offset=1):
    """
    This function implements the computation of the empirical e-values
    """
    evals = []
    n_features = test_score.size

    if offset not in (0, 1):
        raise ValueError("'offset' must be either 0 or 1")

    ko_thr = _coef_diff_threshold(test_score, fdr=fdr, offset=offset)

    for i in range(n_features):
        if test_score[i] < ko_thr:
            evals.append(0)
        else:
            evals.append(n_features / (offset + np.sum(test_score <= -ko_thr)))

    return np.array(evals)
