"""
Test the scenario module
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from hidimstat.scenario import multivariate_1D_simulation
from hidimstat.scenario import multivariate_simulation
from hidimstat.scenario import multivariate_temporal_simulation

ROI_SIZE_2D = 2
SHAPE_2D = (12, 12)

ROI_SIZE_3D = 2
SHAPE_3D = (12, 12, 12)


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
    sigma = 1.0

    # Test 1
    X, y, beta, noise = multivariate_1D_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        sigma=sigma,
        rho=rho,
        shuffle=False,
        seed=0,
    )

    sigma_hat = np.std(noise)
    rho_hat = np.corrcoef(X[:, 19], X[:, 20])[0, 1]

    assert_almost_equal(sigma_hat, sigma, decimal=1)
    assert_almost_equal(rho_hat, rho, decimal=1)
    assert_equal(X.shape, (n_samples, n_features))
    assert_equal(np.count_nonzero(beta), support_size)
    assert_equal(y, np.dot(X, beta) + noise)

    # Test 2
    X, y, beta, noise = multivariate_1D_simulation()
    rho_hat = np.corrcoef(X[:, 19], X[:, 20])[0, 1]
    assert_almost_equal(rho_hat, 0, decimal=1)


def test_multivariate_simulation():
    """Test if the data has expected shape, if the input parameters
    are close to their empirical estimators, if the support has the
    expected size (from simple geometry) and if the noise model is
    the generative model. First test concerns a simulation with a 2D
    structure, second test concerns a simulation with a 3D structure."""

    # Test 1
    n_samples = 100
    shape = SHAPE_2D
    roi_size = ROI_SIZE_2D
    sigma = 1.0
    smooth_X = 1.0
    rho_expected = 0.8
    return_shaped_data = True

    X, y, beta, noise, X_, w = multivariate_simulation(
        n_samples=n_samples,
        shape=shape,
        roi_size=roi_size,
        sigma=sigma,
        smooth_X=smooth_X,
        return_shaped_data=return_shaped_data,
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
    shape = SHAPE_3D
    roi_size = ROI_SIZE_3D
    return_shaped_data = False

    X, y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        shape=shape,
        roi_size=roi_size,
        return_shaped_data=return_shaped_data,
        seed=0,
    )

    assert_equal(X.shape, (n_samples, shape[0] * shape[1] * shape[2]))
    assert_equal(np.count_nonzero(beta), 5 * (roi_size**3))


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
    sigma = 1.0
    rho_noise = 0.9
    rho_data = 0.95

    # Test 1
    X, Y, beta, noise = multivariate_temporal_simulation(
        n_samples=n_samples,
        n_features=n_features,
        n_times=n_times,
        support_size=support_size,
        sigma=sigma,
        rho_noise=rho_noise,
        rho_data=rho_data,
    )

    sigma_hat = np.std(noise[:, -1])
    rho_noise_hat = np.corrcoef(noise[:, -1], noise[:, -2])[0, 1]

    assert_almost_equal(sigma_hat, sigma, decimal=1)
    assert_almost_equal(rho_noise_hat, rho_noise, decimal=1)
    assert_equal(X.shape, (n_samples, n_features))
    assert_equal(Y.shape, (n_samples, n_times))
    assert_equal(np.count_nonzero(beta), support_size * n_times)
    assert_equal(Y, np.dot(X, beta) + noise)

    # Test 2
    X, Y, beta, noise = multivariate_temporal_simulation(
        n_samples=n_samples,
        n_features=n_features,
        n_times=n_times,
        support_size=support_size,
        sigma=sigma,
        rho_noise=rho_noise,
        rho_data=rho_data,
        shuffle=False,
    )

    rho_data_hat = np.corrcoef(X[:, 19], X[:, 20])[0, 1]
    assert_almost_equal(rho_data_hat, rho_data, decimal=1)
    assert_equal(Y, np.dot(X, beta) + noise)
