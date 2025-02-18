"""
Test the adaptive_permutation_threshold module
"""

import numpy as np
from numpy.testing import assert_almost_equal

from hidimstat.ada_svr import ada_svr, ada_svr_pvalue
from hidimstat.scenario import multivariate_1D_simulation


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
    pval, pval_corr, _, _ = ada_svr_pvalue(beta_hat, scale_hat)

    # Check that the p-values are close to 0.5 for the features not in the support
    # and close to 0 for the feature in the support
    expected = 0.5 * np.ones(n_features)
    expected[:support_size] = 0.0

    assert_almost_equal(pval[:support_size], expected[:support_size], decimal=1)
    assert_almost_equal(pval_corr[support_size:], expected[support_size:], decimal=1)
