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
    """
    Test desparsified lasso on a simple simulation with no structure and
    a support of size 5.
     - Test that the confidence intervals contain the true beta
     - Test that the p-values are lower than 0.05 for the important features
       and higher than 0.2 for the non-important features.
    """

    n_samples, n_features = 400, 40
    support_size = 5
    signal_noise_ratio = 32
    rho = 0.0
    confidence = 0.9
    alpha = 1 - confidence

    X, y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        signal_noise_ratio=signal_noise_ratio,
        rho=rho,
        shuffle=False,
    )

    beta_hat, sigma_hat, precision_diag = desparsified_lasso(X, y)
    _, pval_corr, _, _, cb_min, cb_max = desparsified_lasso_pvalue(
        X.shape[0], beta_hat, sigma_hat, precision_diag, confidence=confidence
    )
    # Check that beta is within the confidence intervals

    tolerance = 0.1
    assert np.all(beta >= cb_min - tolerance)
    assert np.all(beta <= cb_max + tolerance)

    # Check p-values for important and non-important features
    important = beta != 0
    non_important = beta == 0
    tp = np.sum(pval_corr[important] < alpha)
    fp = np.sum(pval_corr[non_important] < alpha)
    assert fp / np.sum(non_important) <= alpha
    assert tp / np.sum(important) >= 0.8

    beta_hat, sigma_hat, precision_diag = desparsified_lasso(X, y, dof_ajdustement=True)
    _, pval_corr, _, _, cb_min, cb_max = desparsified_lasso_pvalue(
        X.shape[0], beta_hat, sigma_hat, precision_diag, confidence=confidence
    )
    # Check that beta is within the confidence intervals
    assert np.all(beta >= cb_min - tolerance)
    assert np.all(beta <= cb_max + tolerance)
    # Check p-values for important and non-important features
    tp = np.sum(pval_corr[important] < alpha)
    fp = np.sum(pval_corr[non_important] < alpha)
    assert fp / np.sum(non_important) <= alpha
    assert tp / np.sum(important) >= 0.8


def test_desparsified_group_lasso():
    """Testing the procedure on a simulation with no structure and
    a support of size 2.
    Computing one-sided p-values, we want
    low p-values for the features of the support and p-values
    close to 0.5 for the others."""

    n_samples = 400
    n_features = 40
    n_target = 10
    support_size = 5
    signal_noise_ratio = 32
    rho_serial = 0.9
    alpha = 0.1

    corr = toeplitz(np.geomspace(1, rho_serial ** (n_target - 1), n_target))

    X, Y, beta, _ = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_target,
        support_size=support_size,
        rho_serial=rho_serial,
        signal_noise_ratio=signal_noise_ratio,
    )

    beta_hat, theta_hat, precision_diag = desparsified_lasso(
        X, Y, multioutput=True, covariance=corr
    )
    _, pval_corr, _, _ = desparsified_group_lasso_pvalue(
        beta_hat, theta_hat, precision_diag
    )

    important = beta[:, 0] != 0
    non_important = beta[:, 0] == 0

    assert_almost_equal(beta_hat, beta, decimal=1)
    tp = np.sum(pval_corr[important] < alpha)
    fp = np.sum(pval_corr[non_important] < alpha)
    assert fp / np.sum(non_important) <= alpha
    assert tp / np.sum(important) >= 0.8

    beta_hat, theta_hat, precision_diag = desparsified_lasso(X, Y, multioutput=True)
    _, pval_corr, _, _ = desparsified_group_lasso_pvalue(
        beta_hat, theta_hat, precision_diag
    )

    assert_almost_equal(beta_hat, beta, decimal=1)
    tp = np.sum(pval_corr[important] < alpha)
    fp = np.sum(pval_corr[non_important] < alpha)
    assert fp / np.sum(non_important) <= alpha
    assert tp / np.sum(important) >= 0.8

    # Testing error is raised when the covariance matrix has wrong shape
    bad_cov = np.delete(corr, 0, axis=1)
    np.testing.assert_raises(
        ValueError, desparsified_lasso, X=X, y=Y, multioutput=True, covariance=bad_cov
    )

    with pytest.raises(ValueError, match="Unknown test 'r2'"):
        desparsified_group_lasso_pvalue(beta_hat, theta_hat, precision_diag, test="r2")
