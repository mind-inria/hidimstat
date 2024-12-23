import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from sklearn.utils import _safe_indexing

from hidimstat.stat_tools import pval_from_two_sided_pval_and_sign


def permutation_test_cv(
    X,
    y,
    n_permutations=1000,
    C=None,
    Cs=np.logspace(-7, 1, 9),
    seed=0,
    n_jobs=1,
    verbose=1,
):
    """Cross-validated permutation test shuffling the target

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data.

    y : ndarray, shape (n_samples,)
        Target.

    C : float or None, optional (default=None)
        If None, the linear SVR regularization parameter is set by cross-val
        running a grid search on the list of hyper-parameters contained in Cs.
        Otherwise, the regularization parameter is equal to C.
        The strength of the regularization is inversely proportional to C.

    Cs : ndarray, optional (default=np.logspace(-7, 1, 9))
        If C is None, the linear SVR regularization parameter is set by
        cross-val running a grid search on the list of hyper-parameters
        contained in Cs.

    n_permutations : int, optional (default=1000)
        Number of permutations used to compute the survival function
        and cumulative distribution function scores.

    seed : int, optional (default=0)
        Determines the permutations used for shuffling the target

    n_jobs : int or None, optional (default=1)
        Number of CPUs to use during the cross validation.

    verbose: int, optional (default=1)
        The verbosity level: if non zero, progress messages are printed
        when computing the permutation stats in parralel.
        The frequency of the messages increases with the verbosity level.

    Returns
    -------
    pval_corr : ndarray, shape (n_features,)
        p-value corrected for multiple testing, with numerically accurate
        values for positive effects (ie., for p-value close to zero).

    one_minus_pval_corr : ndarray, shape (n_features,)
        One minus the corrected p-value, with numerically accurate
        values for negative effects (ie., for p-value close to one).
    """

    if C is None:

        steps = [("SVR", LinearSVR())]
        pipeline = Pipeline(steps)
        parameters = {"SVR__C": Cs}
        grid = GridSearchCV(pipeline, param_grid=parameters, n_jobs=n_jobs)
        grid.fit(X, y)
        C = grid.best_params_["SVR__C"]
        estimator = LinearSVR(C=C)

    else:

        estimator = LinearSVR(C=C)

    pval_corr, one_minus_pval_corr = permutation_test(
        X,
        y,
        estimator,
        n_permutations=n_permutations,
        seed=seed,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    return pval_corr, one_minus_pval_corr


def permutation_test(X, y, estimator, n_permutations=1000, seed=0, n_jobs=1, verbose=1, proba=False):
    """Permutation test shuffling the target

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data.

    y : ndarray, shape (n_samples,)
        Target.

    n_permutations : int, optional (default=1000)
        Number of permutations used to compute the survival function
        and cumulative distribution function scores.

    seed : int, optional (default=0)
        Determines the permutations used for shuffling the target

    n_jobs : int or None, optional (default=1)
        Number of CPUs to use during the cross validation.

    verbose: int, optional (default=1)
        The verbosity level: if non zero, progress messages are printed
        when computing the permutation stats in parralel.
        The frequency of the messages increases with the verbosity level.

    Returns
    -------
    pval_corr : ndarray, shape (n_features,)
        p-value corrected for multiple testing, with numerically accurate
        values for positive effects (ie., for p-value close to zero).

    one_minus_pval_corr : ndarray, shape (n_features,)
        One minus the corrected p-value, with numerically accurate
        values for negative effects (ie., for p-value close to one).
    """

    rng = np.random.default_rng(seed)

    stat = _permutation_test_stat(clone(estimator), X, y)

    permutation_stats = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_permutation_test_stat)(clone(estimator), X, _shuffle(y, rng))
        for _ in range(n_permutations)
    )

    permutation_stats = np.array(permutation_stats)
    if proba:
        return permutation_stats[:,0,:]
    two_sided_pval_corr = step_down_max_T(stat, permutation_stats)

    stat_sign = np.sign(stat)

    pval_corr, _, one_minus_pval_corr, _ = pval_from_two_sided_pval_and_sign(
        two_sided_pval_corr, stat_sign
    )

    return pval_corr, one_minus_pval_corr


def _permutation_test_stat(estimator, X, y):
    """Fit estimator and get coef"""
    stat = estimator.fit(X, y).coef_
    return stat


def _shuffle(y, rng):
    """Shuffle vector"""
    indices = rng.permutation(len(y))
    return _safe_indexing(y, indices)


def step_down_max_T(stat, permutation_stats):
    """Step-down maxT algorithm for computing adjusted p-values

    Parameters
    ----------
    stat : ndarray, shape (n_features,)
        Statistic computed on the original (unpermutted) problem.

    permutation_stats : ndarray, shape (n_permutations, n_features)
        Statistics computed on permutted problems.

    Returns
    -------
    two_sided_pval_corr : ndarray, shape (n_features,)
        Two-sided p-values corrected for multiple testing.

    References
    ----------
    .. [1] Westfall, P. H., & Young, S. S. (1993). Resampling-based multiple
           testing: Examples and methods for p-value adjustment (Vol. 279).
           John Wiley & Sons.
    """

    n_permutations, n_features = np.shape(permutation_stats)

    index_ordered = np.argsort(np.abs(stat))
    stat_ranked = np.empty(n_features)
    stat_ranked[index_ordered] = np.arange(n_features)
    stat_ranked = stat_ranked.astype(int)
    stat_sorted = np.copy(np.abs(stat)[index_ordered])
    permutation_stats_ordered = np.copy(np.abs(permutation_stats)[:, index_ordered])

    for i in range(1, n_features):
        permutation_stats_ordered[:, i] = np.maximum(
            permutation_stats_ordered[:, i - 1], permutation_stats_ordered[:, i]
        )

    two_sided_pval_corr = (
        np.sum(np.less_equal(stat_sorted, permutation_stats_ordered), axis=0)
        / n_permutations
    )

    for i in range(n_features - 1)[::-1]:
        two_sided_pval_corr[i] = np.maximum(
            two_sided_pval_corr[i], two_sided_pval_corr[i + 1]
        )

    two_sided_pval_corr = np.copy(two_sided_pval_corr[stat_ranked])

    return two_sided_pval_corr
