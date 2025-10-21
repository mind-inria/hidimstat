"""
Test the adaptive_permutation_threshold module
"""

import numpy as np
from numpy.testing import assert_almost_equal

from hidimstat.scenario import multivariate_1D_simulation
from hidimstat.stat_tools import pval_from_scale
from hidimstat.adaptive_permutation_threshold import ada_svr


def test_ada_svr():
    '''Testing the procedure on a simulation with no structure and a support
    of size 1. Computing one-sided p-values, we want a low p-value
    for the first feature and p-values close to 0.5 for the others.'''

    n_samples, n_features = 20, 50
    support_size = 1
    sigma = 0.1
    rho = 0.0

    X_init, y, beta, noise = \
        multivariate_1D_simulation(n_samples=n_samples, n_features=n_features,
                                   support_size=support_size, sigma=sigma,
                                   rho=rho, shuffle=False, seed=3)

    y = y - np.mean(y)
    X_init = X_init - np.mean(X_init, axis=0)

    beta_hat, scale_hat = ada_svr(X_init, y)

    pval, pval_corr, _, _ = pval_from_scale(beta_hat, scale_hat)

    expected = 0.5 * np.ones(n_features)
    expected[:support_size] = 0.0

    assert_almost_equal(pval[:support_size], expected[:support_size],
                        decimal=1)
    assert_almost_equal(pval_corr[support_size:], expected[support_size:],
                        decimal=1)
