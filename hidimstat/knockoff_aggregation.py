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
    gamma_min=0.05,
    verbose=False,
    memory=None,
    random_state=None,
):

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
            pvals, gamma=gamma, gamma_min=gamma_min, adaptive=adaptive_aggregation
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
