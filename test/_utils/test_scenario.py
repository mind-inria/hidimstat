"""
Test the scenario module
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from hidimstat._utils.scenario import (
    multivariate_simulation,
    multivariate_simulation_autoregressive,
)


def test_multivariate_simulation():
    """Test if the data has expected shape, if the input parameters
    are close to their empirical estimators, if the support has the
    expected size (from simple geometry) and if the noise model is
    the generative model. First test concerns a simulation with a 2D
    structure, second test concerns a simulation with a 3D structure."""

    # Test 1
    n_samples = 100
    shape = (12, 12)
    roi_size = 2
    sigma = 1.0
    smooth_X = 1.0
    rho_expected = 0.8

    X, y, beta, noise, X_, w = multivariate_simulation(
        n_samples=n_samples,
        shape=shape,
        roi_size=roi_size,
        sigma=sigma,
        smooth_X=smooth_X,
        seed=0,
    )

    sigma_hat = np.std(noise)
    rho_hat = np.corrcoef(X[:, 19], X[:, 20])[0, 1]

    assert_almost_equal(sigma_hat, sigma, decimal=1)
    assert_almost_equal(rho_hat, rho_expected, decimal=2)
    assert_equal(X.shape, (n_samples, shape[0] * shape[1]))
    assert_equal(X_.shape, (n_samples, shape[0], shape[1]))
    assert_equal(np.count_nonzero(beta), 4 * (roi_size**2))
    assert_equal(y, np.dot(X, beta) + noise)

    # Test 2
    shape = (12, 12, 12)
    roi_size = 2

    X, y, beta, noise, _, _ = multivariate_simulation(
        n_samples=n_samples,
        shape=shape,
        roi_size=roi_size,
        seed=0,
    )

    assert_equal(X.shape, (n_samples, shape[0] * shape[1] * shape[2]))
    assert_equal(np.count_nonzero(beta), 5 * (roi_size**3))


def test_multivariate_1D_simulation():
    """Test if the data has expected shape, if the input parameters
    are close to their empirical estimators, if the support size is
    correct and if the noise model is the generative model. The
    first test concerns a simulation with a 1D spatial structure,
    the second test concerns a simulation with a random structure"""

    n_samples = 100
    n_features = 500
    support_size = 10
    rho = 0.7
    sigma = 3.0

    # Test 1
    X, y, beta, non_zero, noise_mag, eps = multivariate_simulation_autoregressive(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        sigma_noise=sigma,
        rho=rho,
        shuffle=False,
        seed=42,
    )
    noise = noise_mag * eps

    sigma_hat = np.std(eps)
    rho_hat = np.corrcoef(X[:, 19], X[:, 20])[0, 1]

    assert_almost_equal(sigma_hat, sigma, decimal=0)
    assert_almost_equal(rho_hat, rho, decimal=1)
    assert_equal(X.shape, (n_samples, n_features))
    assert_equal(np.count_nonzero(beta), support_size)
    assert_equal(y, np.dot(X, beta) + noise)

    # Test 2
    X, y, beta, non_zero, noise_mag, eps = multivariate_simulation_autoregressive(
        n_samples=100, n_features=500, rho=0.0
    )
    rho_hat = np.corrcoef(X[:, 19], X[:, 20])[0, 1]
    assert_almost_equal(rho_hat, 0, decimal=1)


def test_multivariate_temporal_simulation():
    """Test if the data has expected shape, if the input parameters
    are close to their empirical estimators, if the support size is
    correct and if the noise model is the generative model. The
    first test concerns a simulation with a 1D spatial structure
    and a temporal structure, the second test concerns a simulation
    with a random spatial structure and a temporal structure."""

    n_samples = 30
    n_features = 50
    n_times = 10
    support_size = 2
    sigma = 3.0
    rho_noise = 0.9
    rho_data = 0.95

    # Test 1
    X, Y, beta, non_zero, noise_mag, eps = multivariate_simulation_autoregressive(
        n_samples=n_samples,
        n_features=n_features,
        n_times=n_times,
        support_size=support_size,
        rho=rho_data,
        sigma_noise=sigma,
        rho_noise_time=rho_noise,
    )
    noise = noise_mag * eps

    sigma_hat = np.std(eps[:, -1])
    rho_noise_hat = np.corrcoef(noise[:, -1], noise[:, -2])[0, 1]

    assert_almost_equal(sigma_hat, sigma, decimal=0)
    assert_almost_equal(rho_noise_hat, rho_noise, decimal=1)
    assert_equal(X.shape, (n_samples, n_features))
    assert_equal(Y.shape, (n_samples, n_times))
    assert_equal(np.count_nonzero(beta), support_size * n_times)
    assert_equal(Y, np.dot(X, beta) + noise)

    # Test 2
    X, Y, beta, non_zero, noise_mag, eps = multivariate_simulation_autoregressive(
        n_samples=n_samples,
        n_features=n_features,
        n_times=n_times,
        rho=rho_data,
        support_size=support_size,
        sigma_noise=sigma,
        rho_noise_time=rho_noise,
        shuffle=False,
    )
    noise = noise_mag * eps

    rho_data_hat = np.corrcoef(X[:, 19], X[:, 20])[0, 1]
    assert_almost_equal(rho_data_hat, rho_data, decimal=1)
    assert_equal(Y, np.dot(X, beta) + noise)


def test_simu_data():
    """Test multivariate 1D simulation AR"""
    X, y, _, _, _, _ = multivariate_simulation_autoregressive(
        n_samples=100, n_features=200, seed=42
    )

    assert X.shape == (100, 200)
    assert y.size == 100


def test_non_zero_index():
    """Test to make sure non-null variable indices are sampled without replacement"""
    X, y, _, non_zero, _, _ = multivariate_simulation_autoregressive(
        10, 10, support_size=10, seed=0
    )
    assert non_zero.size == np.unique(non_zero).size
