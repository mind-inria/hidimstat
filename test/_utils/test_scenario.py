"""
Test the scenario module
"""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal, assert_raises

from hidimstat._utils.scenario import (_generate_2D_weight,
                                       _generate_3D_weight,
                                       multivariate_simulation,
                                       multivariate_simulation_spatial)


@pytest.mark.parametrize(
    "n_samples,shape,roi_size,signal_noise_ratio,smooth_X,rho_expected,seed",
    [
        (100, (12, 12), 2, 1.0, 1.0, 0.8, 0),
        (100, (12, 12), 2, 0.0, 1.0, 0.8, 0),
        (100, (12, 12), 2, np.inf, 1.0, 0.8, 0),
        (100, (12, 12), 0, 1.0, 1.0, 0.8, 0),
    ],
    ids=[
        "basic case",
        "only noise",
        "no noise",
        "no roi",
    ],
)
def test_multivariate_simulation_2D(
    n_samples, shape, roi_size, signal_noise_ratio, smooth_X, rho_expected, seed
):
    """Test concerns a simulation with a 2D
    if the data has expected shape,
    if the input parameters are close to their empirical estimators,
    if the support has the expected size (from simple geometry)
    and if the noise model is the generative model.
    """
    X, y, beta, noise = multivariate_simulation_spatial(
        n_samples=n_samples,
        shape=shape,
        roi_size=roi_size,
        signal_noise_ratio=signal_noise_ratio,
        smooth_X=smooth_X,
        seed=seed,
    )

    signal_noise_ratio_hat = np.linalg.norm(y - noise) / np.linalg.norm(noise)
    rho_hat = np.corrcoef(X[:, 19], X[:, 20])[0, 1]

    # check if the data has expected shape
    assert_equal(X.shape, (n_samples, shape[0] * shape[1]))
    # check if the input parameters are close to their empirical estimators
    if roi_size == 0:
        assert_raises(
            AssertionError,
            assert_almost_equal,
            signal_noise_ratio_hat,
            signal_noise_ratio,
        )
    else:
        assert_almost_equal(signal_noise_ratio_hat, signal_noise_ratio, decimal=1)
    assert_almost_equal(rho_hat, rho_expected, decimal=2)
    if signal_noise_ratio == 0:
        assert_equal(y, noise)
    else:
        assert_equal(y, np.dot(X, beta) + noise)

    # check if the support has the expected size (from simple geometry)
    if roi_size != 0:
        assert_equal(np.count_nonzero(beta), 4 * (roi_size**2))


def test_multivariate_simulation_3D():
    """Test concerns a simulation with a 3D
    if the data has expected shape,
    if the input parameters are close to their empirical estimators,
    if the support has the expected size (from simple geometry)
    and if the noise model is the generative model.
    """
    n_samples = 100
    shape = (12, 12, 12)
    roi_size = 2
    signal_noise_ratio = 1.0
    smooth_X = 1.0
    rho_expected = 0.8

    X, y, beta, noise = multivariate_simulation_spatial(
        n_samples=n_samples,
        shape=shape,
        roi_size=roi_size,
        signal_noise_ratio=signal_noise_ratio,
        smooth_X=smooth_X,
        seed=0,
    )

    signal_noise_ratio_hat = np.linalg.norm(y - noise) / np.linalg.norm(noise)
    rho_hat = np.corrcoef(X[:, 100], X[:, 101])[0, 1]

    # check if the data has expected shape
    assert_equal(X.shape, (n_samples, shape[0] * shape[1] * shape[2]))
    # check if the input parameters are close to their empirical estimators
    assert_almost_equal(signal_noise_ratio_hat, signal_noise_ratio, decimal=1)
    assert_almost_equal(rho_hat, rho_expected, decimal=2)
    assert_equal(y, np.dot(X, beta) + noise)
    # check if the support has the expected size (from simple geometry)
    assert_equal(np.count_nonzero(beta), 5 * (roi_size**3))


def test_multivariate_simulation_edge_cases_2D():
    """Test minimum valid shape and roi_size"""
    shape = (2, 2)
    shape_weight = shape + (5,)
    roi_size = 1
    seed = 42
    n_samples = 6
    w = _generate_2D_weight(shape=shape, roi_size=roi_size)
    assert_equal(w.shape, shape_weight)
    X, y, beta, noise = multivariate_simulation_spatial(
        n_samples=n_samples, shape=shape, roi_size=roi_size, seed=seed
    )
    assert_equal(X.shape, (n_samples, np.prod(shape)))


def test_multivariate_simulation_edge_cases_3D():
    """Test 3D minimum case"""
    shape = (2, 2, 2)
    shape_weight = shape + (5,)
    roi_size = 1
    seed = 42
    n_samples = 6
    w = _generate_3D_weight(shape=shape, roi_size=roi_size)
    assert_equal(w.shape, shape_weight)
    X, y, beta, noise = multivariate_simulation_spatial(
        n_samples=n_samples, shape=shape, roi_size=roi_size, seed=seed
    )
    assert_equal(X.shape, (n_samples, np.prod(shape)))


def test_multivariate_simulation_edge_cases_roi_full():
    """Test roi_size equal to shape dimension"""
    shape = (4, 4)
    shape_weight = shape + (5,)
    roi_size = 4
    seed = 42
    n_samples = 10
    w = _generate_2D_weight(shape=shape, roi_size=roi_size)
    assert_equal(w.shape, shape_weight)
    # all the corner are full
    for i in range(4):
        assert np.all(w[:, :, i].sum() == 16)  # Full coverage of corners
    # only the background is empty
    assert np.all(w[:, :, 4].sum() == 0)  # Full coverage of corners
    X, y, beta, noise = multivariate_simulation_spatial(
        n_samples=n_samples, shape=shape, roi_size=roi_size, seed=seed
    )
    assert_equal(X.shape, (n_samples, np.prod(shape)))


def test_multivariate_simulation_invalid():
    """Test invalid inputs"""
    # Invalid shape dimension
    with pytest.raises(ValueError, match="only 2D and 3D are supported"):
        multivariate_simulation_spatial(shape=(2,))

    with pytest.raises(ValueError, match="only 2D and 3D are supported"):
        multivariate_simulation_spatial(shape=(2, 2, 2, 2))

    # ROI size larger than shape
    with pytest.raises(AssertionError, match="roi_size should be lower than"):
        multivariate_simulation_spatial(shape=(4, 4), roi_size=5)

    # Invalid n_samples
    with pytest.raises(AssertionError, match="n_samples must be strictly positive"):
        multivariate_simulation_spatial(n_samples=0)


def test_multivariate_simulation_reproducibility():
    """Test reproducibility with same seed"""

    params = {"n_samples": 10, "shape": (6, 6), "roi_size": 2, "seed": 42}

    X1, y1, beta1, noise1 = multivariate_simulation_spatial(**params)
    X2, y2, beta2, noise2 = multivariate_simulation_spatial(**params)

    assert_equal(X1, X2)
    assert_equal(y1, y2)
    assert_equal(beta1, beta2)
    assert_equal(noise1, noise2)


def test_multivariate_simulation_weights_2D():
    """Test weight map generation and properties"""

    # 2D weights
    shape = (6, 6)
    roi_size = 2
    w = _generate_2D_weight(shape=shape, roi_size=roi_size)

    # Test ROI locations
    assert np.all(w[0:roi_size, 0:roi_size, 0] == 1.0)  # Upper left
    assert np.all(w[-roi_size:, -roi_size:, 1] == 1.0)  # Lower right
    assert np.all(w[0:roi_size, -roi_size:, 2] == 1.0)  # Upper right
    assert np.all(w[-roi_size:, 0:roi_size, 3] == 1.0)  # Lower left
    assert np.all(w[:, :, 4] == 0.0)  # Background


def test_multivariate_simulation_weights_3D():
    """Test weight map generation and properties"""
    # 3D weights
    shape = (6, 6, 6)
    roi_size = 2
    w = _generate_3D_weight(shape=shape, roi_size=roi_size)

    # Test center ROI location
    center_slice = w[2:4, 2:4, 2:4, 4]
    assert np.all(center_slice == 1.0)

    # Test corner ROI signs
    assert np.all(w[0:roi_size, 0:roi_size, 0:roi_size, 0] == -1.0)
    assert np.all(w[-roi_size:, -roi_size:, 0:roi_size, 1] == 1.0)


@pytest.mark.parametrize(
    "n_samples,n_features,n_targets,support_size,rho,rho_noise,signal_noise_ratio,seed,shuffle",
    [
        # Test case: Basic correlation test
        (100, 500, None, 10, 0.7, None, 3.0, 0, False),
        # Test case: No correlation test
        (100, 500, None, 10, 0.0, None, 3.0, 1, False),
        # Test case: No correlation test
        (100, 500, None, 10, 0.0, None, 3.0, 2, True),
        # Test case: Temporal simulation with noise, no shuffle
        (30, 50, 10, 2, 0.95, 0.9, 3.0, 3, False),
        # Test case: Temporal simulation with noise, no shuffle
        (30, 50, 10, 2, 0.0, 0.9, 3.0, 7, False),
        # Test case: Temporal simulation with noise, with shuffle
        (30, 50, 10, 2, 0.0, 0.9, 3.0, 5, True),
    ],
    ids=[
        "basic_correlation",
        "no_correlation",
        "no_correlation_with_shuffle",
        "temporal_correlation",
        "temporal_no_shuffle",
        "temporal_with_shuffle",
    ],
)
def test_multivariate_simulation_all(
    n_samples,
    n_features,
    n_targets,
    support_size,
    rho,
    rho_noise,
    signal_noise_ratio,
    seed,
    shuffle,
):
    """Test multivariate autoregressive simulation with various configurations"""

    # Create simulation
    params = {
        "n_samples": n_samples,
        "n_features": n_features,
        "support_size": support_size,
        "rho": rho,
        "seed": seed,
        "shuffle": shuffle,
        "signal_noise_ratio": signal_noise_ratio,
    }

    if n_targets is not None:
        params.update(
            {
                "n_targets": n_targets,
                "rho_serial": rho_noise,
            }
        )

    X, y, beta, noise = multivariate_simulation(**params)
    # assertion on the shape of the data
    assert X.shape == (n_samples, n_features)
    assert y.shape[0] == n_samples
    assert beta.shape[0] == n_features

    # Common assertions
    signal_noise_ratio_estimated = np.linalg.norm(y - noise) / np.linalg.norm(noise)
    rho_hat = np.corrcoef(X[:, 19], X[:, 20])[0, 1]

    assert_almost_equal(signal_noise_ratio_estimated, signal_noise_ratio, decimal=0)
    if n_targets is None:
        sigma_hat = np.std(noise)
        assert_almost_equal(sigma_hat, signal_noise_ratio, decimal=0)
    assert_almost_equal(rho_hat, rho, decimal=1)
    assert_equal(X.shape, (n_samples, n_features))

    if n_targets is None:
        # Non-temporal case
        assert_equal(np.count_nonzero(beta), support_size)
        assert_equal(y, np.dot(X, beta) + noise)
    else:
        # assertion on the shape of the data
        assert beta.shape[1] == n_targets
        assert y.shape[1] == n_targets
        # Temporal case
        assert_equal(y.shape, (n_samples, n_targets))
        assert_equal(np.count_nonzero(beta), support_size * n_targets)
        assert_equal(y, np.dot(X, beta) + noise)

        # Additional temporal assertions
        if rho_noise is not None:
            rho_noise_hat = np.corrcoef(noise[:, -1], noise[:, -2])[0, 1]
            assert_almost_equal(rho_noise_hat, rho_noise, decimal=1)


def test_multivariate_simulation_zero_support():
    """Test autoregressive simulation with zero support size."""
    X, y, beta, noise = multivariate_simulation(
        n_samples=50, n_features=100, support_size=0, seed=42
    )
    assert_equal(np.count_nonzero(beta), 0)


def test_multivariate_simulation_zero_signal_noise_ratio():
    """Test autoregressive simulation with zero SNR."""
    X, y, beta, noise = multivariate_simulation(
        n_samples=50, n_features=100, signal_noise_ratio=0.0, seed=42
    )
    assert_equal(y, noise)
    assert_almost_equal(np.std(y - noise), 0.0)


def test_multivariate_simulation_minimal():
    """Test autoregressive simulation with minimal dimensions."""
    X, y, beta, noise = multivariate_simulation(
        n_samples=2, n_features=2, n_targets=2, support_size=1, seed=42
    )
    assert_equal(X.shape, (2, 2))
    assert_equal(y.shape, (2, 2))
    assert_equal(beta.shape, (2, 2))
    assert_equal(beta, [[True, True], [False, False]])


def test_multivariate_simulation_ar_support_size():
    """Test support_size validation."""
    with pytest.raises(
        AssertionError, match="support_size cannot be larger than n_features"
    ):
        multivariate_simulation(n_samples=10, n_features=5, support_size=10, seed=42)


def test_multivariate_simulation_ar_rho():
    """Test rho validation."""
    with pytest.raises(AssertionError, match="rho must be between -1 and 1"):
        multivariate_simulation(n_samples=10, n_features=20, rho=1.5, seed=42)


def test_multivariate_simulation_ar_rho_noise():
    """Test rho_serial validation."""
    with pytest.raises(AssertionError, match="rho_serial must be between -1 and 1"):
        multivariate_simulation(
            n_samples=10, n_features=20, n_targets=5, rho_serial=1.2, seed=42
        )


def test_multivariate_simulation_ar_snr():
    """Test signal noise ratio (signal_noise_ratio) validation."""
    with pytest.raises(AssertionError, match="signal_noise_ratio must be positive"):
        multivariate_simulation(
            n_samples=10, n_features=20, signal_noise_ratio=-1.0, seed=42
        )


def test_multivariate_simulation_ar_n_samples():
    """Test n_samples validation."""
    with pytest.raises(AssertionError, match="n_samples must be positive"):
        multivariate_simulation(n_samples=0, n_features=20, seed=42)


def test_multivariate_simulation_ar_n_features():
    """Test n_features validation."""
    with pytest.raises(AssertionError, match="n_features must be positive"):
        multivariate_simulation(n_samples=10, n_features=0, seed=42)


def test_multivariate_simulation_ar_n_target():
    """Test n_target validation."""
    with pytest.raises(AssertionError, match="n_target must be positive"):
        multivariate_simulation(n_samples=10, n_features=20, n_targets=0, seed=42)
