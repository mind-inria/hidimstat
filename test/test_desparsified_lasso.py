"""
Test the desparsified_lasso module
"""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from scipy.linalg import toeplitz

from hidimstat._utils.scenario import multivariate_simulation
from hidimstat.desparsified_lasso import (
    desparsified_group_lasso_pvalue,
    desparsified_lasso,
    desparsified_lasso_pvalue,
)


def test_desparsified_lasso():
    """Testing the procedure on a simulation with no structure and
    a support of size 1. Computing 99% confidence bounds and checking
    that they contains the true parameter vector."""

    n_samples, n_features = 52, 50
    support_size = 1
    signal_noise_ratio = 50
    rho = 0.0

    X, y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        signal_noise_ratio=signal_noise_ratio,
        rho=rho,
        shuffle=False,
        seed=10,
    )
    expected_pval_corr = np.ones_like(beta) * 0.5
    expected_pval_corr[beta != 0] = 0.0

    beta_hat, sigma_hat, precision_diag = desparsified_lasso(X, y)
    pval, pval_corr, one_minus_pval, one_minus_pval_corr, cb_min, cb_max = (
        desparsified_lasso_pvalue(
            X.shape[0], beta_hat, sigma_hat, precision_diag, confidence=0.99
        )
    )
    assert_almost_equal(beta_hat, beta, decimal=1)
    assert_equal(cb_min < beta, True)
    assert_equal(cb_max > beta, True)
    assert_almost_equal(pval_corr, expected_pval_corr, decimal=1)

    beta_hat, sigma_hat, precision_diag = desparsified_lasso(X, y, dof_ajdustement=True)
    pval, pval_corr, one_minus_pval, one_minus_pval_corr, cb_min, cb_max = (
        desparsified_lasso_pvalue(
            X.shape[0], beta_hat, sigma_hat, precision_diag, confidence=0.99
        )
    )
    assert_almost_equal(beta_hat, beta, decimal=1)
    assert_equal(cb_min < beta, True)
    assert_equal(cb_max > beta, True)
    assert_almost_equal(pval_corr, expected_pval_corr, decimal=1)


def test_desparsified_group_lasso():
    """Testing the procedure on a simulation with no structure and
    a support of size 2. Computing one-sided p-values, we want
    low p-values for the features of the support and p-values
    close to 0.5 for the others."""

    n_samples = 50
    n_features = 100
    n_target = 10
    support_size = 2
    signal_noise_ratio = 5000
    rho_serial = 0.9
    corr = toeplitz(np.geomspace(1, rho_serial ** (n_target - 1), n_target))

    X, Y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_target,
        support_size=support_size,
        rho_serial=rho_serial,
        signal_noise_ratio=signal_noise_ratio,
        seed=10,
    )

    beta_hat, theta_hat, precision_diag = desparsified_lasso(
        X, Y, multioutput=True, covariance=corr
    )
    pval, pval_corr, one_minus_pval, one_minus_pval_corr = (
        desparsified_group_lasso_pvalue(beta_hat, theta_hat, precision_diag)
    )

    expected_pval_corr = np.ones_like(beta[:, 0]) * 0.5
    expected_pval_corr[beta[:, 0] != 0] = 0.0

    assert_almost_equal(beta_hat, beta, decimal=1)
    assert_almost_equal(pval_corr, expected_pval_corr, decimal=1)

    beta_hat, theta_hat, precision_diag = desparsified_lasso(X, Y, multioutput=True)
    pval, pval_corr, one_minus_pval, one_minus_pval_corr = (
        desparsified_group_lasso_pvalue(beta_hat, theta_hat, precision_diag, test="F")
    )

    assert_almost_equal(beta_hat, beta, decimal=1)
    assert_almost_equal(pval_corr, expected_pval_corr, decimal=1)

    # Testing error is raised when the covariance matrix has wrong shape
    bad_cov = np.delete(corr, 0, axis=1)
    np.testing.assert_raises(
        ValueError, desparsified_lasso, X=X, y=Y, multioutput=True, covariance=bad_cov
    )

    with pytest.raises(ValueError, match="Unknown test 'r2'"):
        desparsified_group_lasso_pvalue(beta_hat, theta_hat, precision_diag, test="r2")
