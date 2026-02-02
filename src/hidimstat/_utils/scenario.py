import numpy as np
from scipy import ndimage
from scipy.linalg import toeplitz


def _generate_2D_weight(shape, roi_size):
    """
    Create a 2D weight map with four ROIs (Regions of Interest) in the corners.

    Parameters
    ----------
    shape : tuple of int (n_x, n_z)
        Shape of the 2D data array for which to generate weights.
        n_x : int
            Size in x dimension
        n_z : int
            Size in z dimension
    roi_size : int
        Size of the edge length of each square ROI region.
        ROIs will be placed in all four corners.

    Returns
    -------
    w : numpy.ndarray, shape (n_x, n_z, 5)
        3D weight map array where:
        - First two dimensions match input shape
        - Third dimension contains 5 channels:
            - Channel 0: Upper left ROI weights
            - Channel 1: Lower right ROI weights
            - Channel 2: Upper right ROI weights
            - Channel 3: Lower left ROI weights
            - Channel 4: Background (all zeros)
        Within each ROI region the weights are set to 1.0, elsewhere 0.0

        Create a 2D weight map with four ROIs
    """
    assert roi_size <= np.min(shape), (
        f"roi_size should be lower than the dimension {np.argmin(shape)} of the array"
    )

    w = np.zeros(shape + (5,))
    w[0:roi_size, 0:roi_size, 0] = 1.0
    w[-roi_size:, -roi_size:, 1] = 1.0
    w[0:roi_size, -roi_size:, 2] = 1.0
    w[-roi_size:, 0:roi_size, 3] = 1.0

    return w


def _generate_3D_weight(shape, roi_size):
    """
    Create a 3D weight map with five ROIs (Regions of Interest) in specific locations.

    Parameters
    ----------
    shape : tuple of int (n_x, n_y, n_z)
        Shape of the 3D data array for which to generate weights.
        n_x : int
            Size in x dimension
        n_y : int
            Size in y dimension
        n_z : int
            Size in z dimension
    roi_size : int
        Size of the edge length of each cubic ROI region.
        ROIs will be placed in corners and center.

    Returns
    -------
    w : numpy.ndarray, shape (n_x, n_y, n_z, 5)
        4D weight map array where:
        - First three dimensions match input shape
        - Fourth dimension contains 5 channels:
            - Channel 0: Front-left-top ROI weights (-1.0)
            - Channel 1: Back-right-top ROI weights (1.0)
            - Channel 2: Front-right-bottom ROI weights (-1.0)
            - Channel 3: Back-left-bottom ROI weights (1.0)
            - Channel 4: Center ROI weights (1.0)
    """
    assert roi_size <= np.min(shape), (
        f"roi_size should be lower than the dimension {np.argmin(shape)} of the array"
    )

    w = np.zeros(shape + (5,))
    w[0:roi_size, 0:roi_size, 0:roi_size, 0] = -1.0
    w[-roi_size:, -roi_size:, 0:roi_size, 1] = 1.0
    w[0:roi_size, -roi_size:, -roi_size:, 2] = -1.0
    w[-roi_size:, 0:roi_size, -roi_size:, 3] = 1.0
    w[
        (shape[0] - roi_size) // 2 : (shape[0] + roi_size) // 2,
        (shape[1] - roi_size) // 2 : (shape[1] + roi_size) // 2,
        (shape[2] - roi_size) // 2 : (shape[2] + roi_size) // 2,
        4,
    ] = 1.0
    return w


def multivariate_simulation_spatial(
    n_samples=100,
    shape=(12, 12),
    roi_size=2,
    signal_noise_ratio=1.0,
    smooth_X=1.0,
    seed=0,
):
    """
    Generate a multivariate simulation with 2D or 3D data.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples.
    shape : tuple of int, default=(12, 12)
        Shape of the data in the simulation. Either (n_x, n_y) for 2D
        or (n_x, n_y, n_z) for 3D data.
    roi_size : int, default=2
        Size of the edge of the ROIs (Regions of Interest).
    signal_noise_ratio : float, default=1.0
        Signal-to-noise ratio. Controls noise scaling.
    smooth_X : float, default=1.0
        Level of data smoothing using a Gaussian filter.
    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Design matrix with n_features = product of shape dimensions.
    y : ndarray of shape (n_samples,)
        Target vector y = X @ beta + noise.
    beta : ndarray of shape (n_features,)
        Parameter vector (flattened weight map).
    noise : ndarray of shape (n_samples,)
        Additive white Gaussian noise vector.
    X_ : ndarray of shape (n_samples,) + shape
        Reshaped design matrix matching input dimensions.
    w : ndarray of shape shape + (5,)
        Weight map with 5 channels for different ROIs.
    """
    assert n_samples > 0, "n_samples must be strictly positive"
    # Setup seed generator
    rng = np.random.default_rng(seed)

    # Generate the variables from normal distribution
    X_ = rng.standard_normal((n_samples,) + shape)
    ## Create local correlation
    X = []
    for i in np.arange(n_samples):
        Xi = ndimage.gaussian_filter(X_[i], smooth_X)
        X.append(Xi.ravel())
    X = np.asarray(X)

    # Generate the support of the data
    if len(shape) == 2:
        w = _generate_2D_weight(shape, roi_size)
    elif len(shape) == 3:
        w = _generate_3D_weight(shape, roi_size)
    else:
        raise ValueError(
            f"Shape {shape} not supported, only 2D and 3D are supported"
        )
    beta = w.sum(-1).ravel()
    prod_temp = np.dot(X, beta)

    # Scale the noise for respecting signal-noise-ratio.
    eps = rng.standard_normal(size=n_samples)
    if roi_size == 0:
        noise_mag = 1.0
    elif np.isinf(signal_noise_ratio):
        noise_mag = 0.0
    elif signal_noise_ratio != 0.0:
        noise_mag = np.linalg.norm(prod_temp) / (
            np.linalg.norm(eps) * np.sqrt(signal_noise_ratio)
        )
    else:
        prod_temp = np.zeros_like(prod_temp)
        noise_mag = 1
    noise = noise_mag * eps

    y = prod_temp + noise

    return X, y, beta, noise


def multivariate_simulation(
    n_samples,
    n_features,
    n_targets=None,
    support_size=10,
    rho=0,
    value=1.0,
    signal_noise_ratio=10.0,
    rho_serial=0.0,
    shuffle=False,
    continuous_support=False,
    seed=None,
):
    """
    Generate a linear simulation with a Toeplitz covariance structure and optional temporal correlation.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of features/variables.
    n_targets : int or None, default=None
        Number of targets/time points. If None, generates univariate target data.
        If integer > 0, generates multivariate target data.
    support_size : int, default=10
        Number of non-zero coefficients in beta.
    rho : float, default=0
        Feature correlation coefficient for Toeplitz covariance matrix.
    value : float, default=1.0
        Value assigned to non-zero coefficients in beta.
    signal_noise_ratio : float, default=10.0
        Signal-to-noise ratio. Controls noise scaling.
    rho_serial : float, default=0.0
        Serial correlation coefficient of the noise.
    shuffle : bool, default=False
        Whether to shuffle features randomly to avoid to have correlated
        adjacent features.
    continuous_support: bool, default=False
        If True, places non-zero coefficients continuously at the start of beta.
        If False, randomly distributes them throughout beta.
    seed : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Design matrix with Toeplitz covariance structure.
    y : ndarray of shape (n_samples,) or (n_samples, n_target)
        Target vector/matrix, generated as y = X @ beta + noise.
    beta_true : ndarray of shape (n_features,) or (n_features, n_target)
        True coefficient vector/matrix with support_size non-zero elements.
    non_zero : ndarray
        Indices of non-zero coefficients in beta_true.
    noise_factor : float
        Applied noise magnitude scaling factor.
    eps : ndarray of shape (n_samples,) or (n_samples, n_target)
        Noise vector/matrix with optional temporal correlation.
    """
    assert n_samples > 0, "n_samples must be positive"
    assert n_features > 0, "n_features must be positive"
    assert n_targets is None or n_targets > 0.0, "n_target must be positive"
    assert support_size <= n_features, (
        "support_size cannot be larger than n_features"
    )
    assert rho >= -1.0 and rho <= 1.0, "rho must be between -1 and 1"
    assert rho_serial >= -1.0 and rho_serial <= 1.0, (
        "rho_serial must be between -1 and 1"
    )
    assert signal_noise_ratio >= 0.0, "signal_noise_ratio must be positive"
    # Setup seed generator
    rng = np.random.default_rng(seed)

    # Generate the variables from a multivariate normal distribution
    mu = np.zeros(n_features)
    covariance_features = toeplitz(
        rho ** np.arange(0, n_features)
    )  # covariance matrix of X
    X = rng.multivariate_normal(mu, covariance_features, size=(n_samples))

    # Optionally shuffle the samples
    if shuffle:
        rng.shuffle(X.T)

    # Generate the response from a linear model
    if continuous_support:
        non_zero = range(support_size)
    else:
        non_zero = rng.choice(n_features, support_size, replace=False)

    # Generate the support and the noise of the data for the prediction.
    if n_targets is None:
        beta_true = np.zeros(n_features, dtype=bool)
        beta_true[non_zero] = value
        eps = rng.standard_normal(size=n_samples)
    else:
        beta_true = np.zeros((n_features, n_targets), dtype=bool)
        beta_true[non_zero, :] = value
        # possibility to generate correlated noise
        covariance_temporal = toeplitz(
            rho_serial ** np.arange(0, n_targets)
        )  # covariance matrix of time
        eps = rng.multivariate_normal(
            np.zeros(n_targets), covariance_temporal, size=(n_samples)
        )
    prod_temp = np.dot(X, beta_true)

    # Scale the noise for respecting signal-noise-ratio.
    if support_size == 0:
        noise_mag = 1.0
    elif np.isinf(signal_noise_ratio):
        noise_mag = 0.0
    elif signal_noise_ratio != 0.0:
        noise_mag = np.linalg.norm(prod_temp) / (
            np.linalg.norm(eps) * np.sqrt(signal_noise_ratio)
        )
    else:
        prod_temp = np.zeros_like(prod_temp)
        noise_mag = 1
    noise = noise_mag * eps

    y = prod_temp + noise

    return X, y, beta_true, noise


def empirical_snr(X, y, beta, noise=None):
    """
    Compute the empirical signal-to-noise ratio (SNR) for
    the linear model y = X @ beta + noise.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix.

    y : ndarray, shape (n_samples,)
        Target vector.

    beta : ndarray, shape (n_features,)
        Parameter vector.

    noise : ndarray, shape (n_samples,), optional
        Noise vector. If None, computed as y - X @ beta.

    Returns
    -------
    signal_noise_ratio_hat : float
        Empirical SNR computed as var(signal) / var(noise).

    Notes
    -----
    SNR measures the ratio of signal power to noise power,
    indicating model estimation quality.
    Higher values suggest better signal recovery.
    """
    X = np.asarray(X)

    signal = np.dot(X, beta)

    if noise is None:
        noise = y - signal

    # compute signal-to-noise ratio
    signal_noise_ratio_ = (np.linalg.norm(signal) / np.linalg.norm(noise)) ** 2

    return signal_noise_ratio_
