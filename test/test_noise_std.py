"""
Test the noise_std module
"""

import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from scipy.linalg import toeplitz

from hidimstat.noise_std import empirical_snr, reid
from hidimstat._utils.scenario import multivariate_simulation


def test_reid():
    """Estimating noise standard deviation in two scenarios.
    First scenario: no structure and a support of size 2.
    Second scenario: no structure and an empty support."""

    n_samples, n_features = 50, 30
    snr = 2.0

    # First expe
    # ##########
    support_size = 2

    X, y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        rho=0.25,
        snr=snr,
        seed=0,
    )

    # max_iter=1 to get a better coverage
    sigma_hat, _ = reid(X, y, tolerance=1e-3, max_iterance=1)
    expected_sigma = snr
    error_relative = np.abs(sigma_hat - expected_sigma) / expected_sigma
    assert error_relative < 0.5

    # Second expe
    # ###########
    support_size = 0

    X, y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        snr=snr,
        seed=2,
    )

    sigma_hat, _ = reid(X, y)
    expected_sigma = snr
    error_relative = np.abs(sigma_hat - expected_sigma) / expected_sigma
    assert error_relative < 0.5


def test_group_reid():
    """Estimating (temporal) noise covariance matrix in two scenarios.
    First scenario: no data structure and a support of size 2.
    Second scenario: no data structure and an empty support."""

    n_samples = 30
    n_features = 50
    n_target = 100
    snr = 3.0
    rho_serial = 0.9

    # First expe
    # ##########
    support_size = 2
    X, Y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_target,
        support_size=support_size,
        snr=snr,
        rho_serial=rho_serial,
        rho=0.0,
        seed=0,
    )
    corr = toeplitz(np.geomspace(1, rho_serial ** (n_target - 1), n_target))

    # max_iter=1 to get a better coverage
    cov_hat, _ = reid(X, Y, multioutput=True, tolerance=1e-3, max_iterance=1)
    error_relative = np.abs(cov_hat - corr) / corr
    assert np.max(error_relative) < 0.3

    cov_hat, _ = reid(X, Y, multioutput=True, method="AR")
    error_relative = np.abs(cov_hat - corr) / corr
    assert np.max(error_relative) < 0.3

    cov_hat, _ = reid(X, Y, multioutput=True, stationary=False)
    error_relative = np.abs(cov_hat - corr) / corr
    assert np.max(error_relative) > 1.0


def test_group_reid_2():
    """Estimating (temporal) noise covariance matrix in two scenarios.
    First scenario: no data structure and a support of size 2.
    Second scenario: no data structure and an empty support."""

    n_samples = 30
    n_features = 50
    n_target = 100
    snr = 1.0
    rho_serial = 0.9

    # Second expe
    # ###########
    support_size = 0
    X, Y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_target,
        rho=0.25,
        support_size=support_size,
        snr=snr,
        rho_serial=rho_serial,
        seed=4,
    )
    # corr = toeplitz(np.geomspace(1, rho_serial ** (n_target - 1), n_target))
    corr = toeplitz(rho_serial ** np.arange(0, n_target))  # covariance matrix of time

    cov_hat, _ = reid(X, Y, multioutput=True)
    error_relative = np.abs(cov_hat - corr) / corr
    assert np.max(error_relative) < 0.2

    cov_hat, _ = reid(X, Y, multioutput=True, method="AR")
    error_relative = np.abs(cov_hat - corr) / corr
    assert np.max(error_relative) < 0.2

    cov_hat, _ = reid(X, Y, multioutput=True, stationary=False)
    error_relative = np.abs(cov_hat - corr) / corr
    assert np.max(error_relative) > 0.5


def test_reid_exception():
    "Test for testing the exceptions on the arguments of reid function"
    n_samples, n_features = 50, 30
    n_target = 10
    snr = 1.0
    rho_serial = 0.9

    # First expe
    # ##########
    support_size = 2

    X, y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_target,
        support_size=support_size,
        snr=snr,
        rho_serial=rho_serial,
    )
    with pytest.raises(
        ValueError, match="Unknown method for estimating the covariance matrix"
    ):
        _, _ = reid(X, y, method="test", multioutput=True)
    with pytest.raises(
        ValueError, match="The AR method is not compatible with the non-stationary"
    ):
        _, _ = reid(X, y, method="AR", stationary=False, multioutput=True)
    with pytest.raises(ValueError, match="The requested AR order is to high with"):
        _, _ = reid(X, y, method="AR", order=1e4, multioutput=True)


def test_empirical_snr():
    """Computing empirical signal to noise ratio in presence of high level of
    noise from the target `y`, the data `X` and the true parameter vector `beta`
    in a simple scenario with a 1D data structure."""

    n_samples, n_features = 30, 30
    support_size = 10
    snr_expected = 0.5

    X, y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        snr=snr_expected,
        seed=0,
    )

    snr = empirical_snr(X, y, beta)

    assert_almost_equal(snr, snr_expected, decimal=2)


def test_empirical_snr_2():
    """Computing empirical signal to noise ratio from the target `y`,
    the data `X` and the true parameter vector `beta` in a simple
    scenario with a 1D data structure."""

    n_samples, n_features = 30, 30
    support_size = 10
    snr_expected = 10.0

    X, y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        snr=snr_expected,
        seed=0,
    )

    snr = empirical_snr(X, y, beta)

    assert_almost_equal(snr, snr_expected, decimal=0)
