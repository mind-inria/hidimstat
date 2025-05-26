import numpy as np
import pytest
from hidimstat._utils.scenario import multivariate_simulation_autoregressive


@pytest.fixture
def data_generator(
    n_samples, n_features, support_size, rho, seed, value, snr, rho_noise_time
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
    snr : float
        Signal-to-noise ratio.
    rho_noise_time : float
        Time correlation coefficient in the noise component.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target vector.
    important_features : ndarray
        Indices of features with non-zero coefficients.
    not_important_features : ndarray
        Indices of features with zero coefficients.
    """
    X, y, beta, _, _, _ = multivariate_simulation_autoregressive(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        rho=rho,
        value=value,
        snr=snr,
        rho_noise_time=rho_noise_time,
        shuffle=False,
        seed=seed,
    )
    important_features = np.where(beta != 0)[0]
    not_important_features = np.where(beta == 0)[0]
    return X, y, important_features, not_important_features
