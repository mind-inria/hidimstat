"""
Test the empirical thresholding module
"""

import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from hidimstat._utils.scenario import multivariate_1D_simulation
from hidimstat.empirical_thresholding import empirical_thresholding
from hidimstat.statistical_tools.utils import pval_from_scale

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor


def test_emperical_thresholding():
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

    beta_hat, scale_hat = empirical_thresholding(X_init, y)

    pval, pval_corr, _, _ = pval_from_scale(beta_hat, scale_hat)

    expected = 0.5 * np.ones(n_features)
    expected[:support_size] = 0.0

    assert_almost_equal(pval_corr, expected, decimal=1)


def test_emperical_thresholding_lasso():
    """Testing the procedure on a simulation with no structure and a support
    of size 1 with lasso."""

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

    with pytest.raises(ValueError, match="linear estimator should be linear."):
        beta_hat, scale_hat = empirical_thresholding(
            X_init, y, linear_estimator=DecisionTreeRegressor()
        )

    beta_hat, scale_hat = empirical_thresholding(X_init, y, linear_estimator=Lasso())

    pval, pval_corr, _, _ = pval_from_scale(beta_hat, scale_hat)

    expected = 0.5 * np.ones(n_features)
    expected[:support_size] = 0.0

    assert_almost_equal(pval_corr, expected, decimal=1)
