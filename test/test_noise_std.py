"""
Test the noise_std module
"""

import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from scipy.linalg import toeplitz

from hidimstat.noise_std import empirical_snr, reid
from hidimstat.scenario import (
    multivariate_1D_simulation,
    multivariate_temporal_simulation,
)


def test_reid():
    """Estimating noise standard deviation in two scenarios.
    First scenario: no structure and a support of size 2.
    Second scenario: no structure and an empty support."""

    n_samples, n_features = 50, 30
    sigma = 2.0

    # First expe
    # ##########
    support_size = 2

    X, y, beta, noise = multivariate_1D_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        sigma=sigma,
        seed=0,
    )

    # max_iter=1 to get a better coverage
    sigma_hat, _ = reid(X, y, tol=1e-3, max_iter=1)
    expected = sigma

    assert_almost_equal(sigma_hat / expected, 1.0, decimal=0)

    # Second expe
    # ###########
    support_size = 0

    X, y, beta, noise = multivariate_1D_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        sigma=sigma,
        seed=1,
    )

    sigma_hat, _ = reid(X, y)
    expected = sigma

    assert_almost_equal(sigma_hat / expected, 1.0, decimal=1)


def test_group_reid():
    """Estimating (temporal) noise covariance matrix in two scenarios.
    First scenario: no data structure and a support of size 2.
    Second scenario: no data structure and an empty support."""

    n_samples = 30
    n_features = 50
    n_times = 10
    sigma = 1.0
    rho = 0.9
    corr = toeplitz(np.geomspace(1, rho ** (n_times - 1), n_times))
    cov = np.outer(sigma, sigma) * corr

    # First expe
    # ##########
    support_size = 2

    X, Y, beta, noise = multivariate_temporal_simulation(
        n_samples=n_samples,
        n_features=n_features,
        n_times=n_times,
        support_size=support_size,
        sigma=sigma,
        rho_noise=rho,
    )

    # max_iter=1 to get a better coverage
    cov_hat, _ = reid(X, Y, group=True, tol=1e-3, max_iter=1)
    error_ratio = cov_hat / cov

    assert_almost_equal(np.max(error_ratio), 1.0, decimal=0)
    assert_almost_equal(np.log(np.min(error_ratio)), 0.0, decimal=1)

    cov_hat, _ = reid(X, Y, group=True, method="AR")
    error_ratio = cov_hat / cov

    assert_almost_equal(np.max(error_ratio), 1.0, decimal=0)
    assert_almost_equal(np.log(np.min(error_ratio)), 0.0, decimal=0)

    cov_hat, _ = reid(X, Y, group=True, stationary=False)
    error_ratio = cov_hat / cov

    assert_almost_equal(np.max(error_ratio), 1.0, decimal=0)
    assert_almost_equal(np.log(np.min(error_ratio)), 0.0, decimal=0)

    # Second expe
    # ###########
    support_size = 0

    X, Y, beta, noise = multivariate_temporal_simulation(
        n_samples=n_samples,
        n_features=n_features,
        n_times=n_times,
        support_size=support_size,
        sigma=sigma,
        rho_noise=rho,
        seed=1,
    )

    cov_hat, _ = reid(X, Y, group=True)
    error_ratio = cov_hat / cov

    assert_almost_equal(np.max(error_ratio), 1.0, decimal=0)
    assert_almost_equal(np.log(np.min(error_ratio)), 0.0, decimal=1)

    cov_hat, _ = reid(X, Y, group=True, method="AR")
    error_ratio = cov_hat / cov

    assert_almost_equal(np.max(error_ratio), 1.0, decimal=0)
    assert_almost_equal(np.log(np.min(error_ratio)), 0.0, decimal=1)


def test_reid_exception():
    "Test for testing the exceptions on the arguments of reid function"
    n_samples, n_features = 50, 30
    n_times = 10
    sigma = 1.0
    rho = 0.9

    # First expe
    # ##########
    support_size = 2

    X, y, beta, noise = multivariate_temporal_simulation(
        n_samples=n_samples,
        n_features=n_features,
        n_times=n_times,
        support_size=support_size,
        sigma=sigma,
        rho_noise=rho,
    )
    with pytest.raises(
        ValueError, match="Unknown method for estimating the covariance matrix"
    ):
        _, _ = reid(X, y, method="test", group=True)
    with pytest.raises(
        ValueError, match="The AR method is not compatible with the non-stationary"
    ):
        _, _ = reid(X, y, method="AR", stationary=False, group=True)
    with pytest.raises(ValueError, match="The requested AR order is to high with"):
        _, _ = reid(X, y, method="AR", order=1e4, group=True)


def test_empirical_snr():
    """Computing empirical signal to noise ratio from the target `y`,
    the data `X` and the true parameter vector `beta` in a simple
    scenario with a 1D data structure."""

    n_samples, n_features = 30, 30
    support_size = 10
    sigma = 2.0

    X, y, beta, noise = multivariate_1D_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        sigma=sigma,
        seed=0,
    )

    snr = empirical_snr(X, y, beta)
    expected = 2.0

    assert_almost_equal(snr, expected, decimal=0)
