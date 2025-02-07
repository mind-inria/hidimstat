"""
Test the permutation test module
"""

import numpy as np
from numpy.testing import assert_almost_equal

from hidimstat.permutation_test import permutation_test_cv
from hidimstat.scenario import multivariate_1D_simulation


def test_permutation_test():
    """Testing the procedure on a simulation with no structure and a support
    of size 1. Computing one-sided p-values, we want a low p-value
    for the first feature and p-values close to 0.5 for the others."""

    n_samples, n_features = 20, 50
    support_size = 1
    sigma = 0.1
    rho = 0.0

    X_init, y, beta, noise = multivariate_1D_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        sigma=sigma,
        rho=rho,
        shuffle=False,
        seed=3,
    )

    y = y - np.mean(y)
    X_init = X_init - np.mean(X_init, axis=0)

    pval_corr, one_minus_pval_corr = permutation_test_cv(X_init, y, n_permutations=100)

    expected = 0.5 * np.ones(n_features)
    expected[:support_size] = 0.0

    assert_almost_equal(pval_corr, expected, decimal=1)


def test_permutation_test_with_SVR():
    """Testing the procedure on a simulation with no structure and a support
    of size 1. Computing one-sided p-values, we want a low p-value
    for the first feature and p-values close to 0.5 for the others."""

    n_samples, n_features = 20, 50
    support_size = 1
    sigma = 0.1
    rho = 0.0

    X_init, y, beta, noise = multivariate_1D_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        sigma=sigma,
        rho=rho,
        shuffle=False,
        seed=3,
    )

    y = y - np.mean(y)
    X_init = X_init - np.mean(X_init, axis=0)

    pval_corr, one_minus_pval_corr = permutation_test_cv(
        X_init, y, n_permutations=100, C=0.1
    )

    expected = 0.5 * np.ones(n_features)
    expected[:support_size] = 0.0

    assert_almost_equal(pval_corr, expected, decimal=1)
