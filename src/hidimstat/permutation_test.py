import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone

from hidimstat.stat_tools import pval_from_two_sided_pval_and_sign, step_down_max_t


def permutation_test(X, y, estimator, n_permutations=1000, seed=0, n_jobs=1, verbose=0):
    """
    Permutation test

    This function compute the distribution of the weights of a linear model
    by shuffling the target :footcite:t:`hirschhorn2005genome`.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data.

    y : ndarray, shape (n_samples,)
        Target.

    estimator : object LinearModel
        The linear model used to fit the data.

    n_permutations : int, optional (default=1000)
        Number of permutations used to compute the survival function
        and cumulative distribution function scores.

    seed : int, optional (default=0)
        Determines the permutations used for shuffling the target

    n_jobs : int or None, optional (default=1)
        Number of CPUs to use during the cross validation.

    verbose : int, optional (default=0)
        The verbosity level of the joblib.Parallel.

    Returns
    -------
    weights : ndarray, shape (n_features,)
        The weights of the original model.

    weights_distribution : ndarray, shape (n_permutations, n_features)
        The distribution of the weights of the model obtained by shuffling
        the target n_permutations times.

    References
    ----------
    .. footbibliography::

    """

    rng = np.random.default_rng(seed)

    # Get the weights of the original model
    if not hasattr(estimator, "coef_"):
        weights = _fit_and_weights(estimator, X, y)
    else:
        weights = estimator.coef_

    # Get the distribution of the weights by shuffling the target
    weights_distribution = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_fit_and_weights)(clone(estimator), X, _shuffle(y, rng))
        for _ in range(n_permutations)
    )

    # Convert the list of weights into an array
    weights_distribution = np.array(weights_distribution)

    return weights, weights_distribution


def permutation_test_pval(weights, weights_distribution):
    """
    Compute p-value from permutation test

    Parameters
    ----------
    weights : ndarray, shape (n_features,)
        The weights of the original model.

    weights_distribution : ndarray, shape (n_permutations, n_features)
        The distribution of the weights of the model obtained by shuffling

    Returns
    -------
    pval_corr : ndarray, shape (n_features,)
        p-value corrected for multiple testing, with numerically accurate
        values for positive effects (ie., for p-value close to zero).

    one_minus_pval_corr : ndarray, shape (n_features,)
        One minus the corrected p-value, with numerically accurate
        values for negative effects (ie., for p-value close to one).
    """
    two_sided_pval_corr = step_down_max_t(weights, weights_distribution)

    stat_sign = np.sign(weights)

    pval_corr, _, one_minus_pval_corr, _ = pval_from_two_sided_pval_and_sign(
        two_sided_pval_corr, stat_sign
    )

    return pval_corr, one_minus_pval_corr


def _fit_and_weights(estimator, X, y):
    """
    Fit the estimator and return the weights

    Parameters
    ----------
    estimator : object
        The estimator to fit.

    X : ndarray, shape (n_samples, n_features)
        Data.

    y : ndarray, shape (n_samples,)
        Target.

    Returns
    -------
    weights : ndarray, shape (n_features,)
        The weights of the estimator.
    """
    weights = estimator.fit(X, y).coef_
    return weights


def _shuffle(y, rng):
    """
    Shuffle the target

    Parameters
    ----------
    y : ndarray, shape (n_samples,)
        Target.

    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    y_shuffled : ndarray, shape (n_samples,)
        Shuffled target.
    """
    y_copy = np.copy(y)
    rng.shuffle(y_copy)
    return y_copy
