import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from hidimstat._utils.scenario import multivariate_simulation

try:
    import matplotlib
except ImportError:
    matplotlib = None


def pytest_configure(config):  # noqa: ARG001
    """Use Agg so that no figures pop up."""
    if matplotlib is not None:
        matplotlib.use("Agg", force=True)


def _rng(seed=42):
    return np.random.default_rng(seed)


@pytest.fixture()
def rng():
    """Return a seeded random number generator."""
    return _rng()


def fitted_linear_regression():
    """Return a fitted linear regression model."""
    X = _rng().integers(0, 2, size=(100, 2, 1))
    estimator = LinearRegression()
    estimator.fit(X[:, 0], X[:, 1])
    return estimator


@pytest.fixture
def data_generator(
    n_samples,
    n_features,
    support_size,
    rho,
    seed,
    value,
    signal_noise_ratio,
    rho_serial,
):
    """
    Generate simulated data for testing.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    n_features : int
        Number of features in the dataset.
    support_size : int
        Number of important features (features with non-zero coefficients).
    rho : float
        Correlation coefficient between features.
    seed : int
        Random seed for reproducibility.
    value : float
        Value to be used for non-zero coefficients.
    signal_noise_ratio : float
        Signal-to-noise ratio.
    rho_serial : float
        Time correlation coefficient in the noise component.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target vector.
    important_features_mask : ndarray
        Mask array of features with non-zero coefficients.
    """
    X, y, beta, _ = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        rho=rho,
        value=value,
        signal_noise_ratio=signal_noise_ratio,
        rho_serial=rho_serial,
        shuffle=False,
        seed=seed,
    )
    important_features_mask = beta.astype(bool)
    return X, y, important_features_mask
