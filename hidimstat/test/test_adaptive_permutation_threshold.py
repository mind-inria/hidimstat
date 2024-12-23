"""
Test the adaptive_permutation_threshold module
"""

import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.svm import SVR

from hidimstat.ada_svr import ada_svr
from hidimstat.scenario import multivariate_1D_simulation
from hidimstat.stat_tools import pval_from_scale
from hidimstat.permutation_test import permutation_test


def test_ada_svr():
    """
    Testing the procedure on a simulation with no structure and a support
    of size 1. Computing one-sided p-values, we want a low p-value
    for the first feature and p-values close to 0.5 for the others.
    """

    # Parameters for the generation of data
    n_samples, n_features = 20, 50
    support_size = 4

    X_init, y, _, _ = multivariate_1D_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        sigma=0.1,
        shuffle=False,
        seed=42,
    )

    # Run the procedure
    beta_hat, scale_hat = ada_svr(X_init, y)

    # Compute p-values
    pval, pval_corr, _, _ = pval_from_scale(beta_hat, scale_hat)

    # Check that the p-values are close to 0.5 for the features not in the support
    # and close to 0 for the feature in the support
    expected = 0.5 * np.ones(n_features)
    expected[:support_size] = 0.0

    assert_almost_equal(pval[:support_size], expected[:support_size], decimal=1)
    assert_almost_equal(pval_corr[support_size:], expected[support_size:], decimal=1)


def test_ada_svr_rcond():
    """
    Testing the effect of rcond
    """
    # create dataset
    X, y, beta, _ = multivariate_1D_simulation(
        n_samples=20,
        n_features=50,
        support_size=3,
        sigma=0.1,
        shuffle=False,
        seed=42,
    )
    X[:10] *= 1e-5
    beta_hat, scale = ada_svr(X, y)
    beta_hat_2, scale_2 = ada_svr(X, y, rcond=1e-15)
    assert np.max(np.abs(beta_hat - beta_hat_2)) > 1
    assert np.max(np.abs(scale - scale_2)) > 1


def test_ada_svr_vs_permutation():
    """
    Validate the adaptive permutation threshold procedure against a permutation
    test. The adaptive permutation threshold procedure should good approciation
    of the proba of the permutation test.
    """
    # create dataset
    X, y, beta, _ = multivariate_1D_simulation(
        n_samples=10,
        n_features=100,
        support_size=1,
        sigma=0.1,
        shuffle=False,
        seed=42,
    )
    beta_hat, scale = ada_svr(X, y)
    # fit a SVR to get the coefficients
    estimator = SVR(kernel="linear", epsilon=0.0, gamma="scale", C=1.0)
    estimator.fit(X, y)
    beta_hat_svr = estimator.coef_

    # compare that the coefficiants are the same that the one of SVR
    assert np.max(np.abs(beta_hat - beta_hat_svr.T[:, 0])) < 2e-4

    proba = permutation_test(
        X, y, estimator=estimator, n_permutations=10000, n_jobs=8, seed=42, proba=True
    )
    assert np.max(np.abs(np.mean(proba, axis=0))) < 1e-3
    assert np.max(np.abs(scale - np.std(proba, axis=0)) / scale) < 1e-1
