import numpy as np
from scipy import ndimage
from scipy.linalg import toeplitz


def _generate_2D_weight(shape, roi_size):
    """
    Create a 2D weight map with four ROIs (Regions of Interest) in the corners.

    Parameters:
    -----------
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


def multivariate_simulation(
    n_samples=100,
    shape=(12, 12),
    roi_size=2,
    sigma=1.0,
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
    sigma : float, default=1.0
        Standard deviation of the additive white Gaussian noise.
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
    # Setup seed generator
    rng = np.random.default_rng(seed)

    # generate the support of the data
    if len(shape) == 2:
        w = _generate_2D_weight(shape, roi_size)
    elif len(shape) == 3:
        w = _generate_3D_weight(shape, roi_size)
    else:
        raise ValueError(f"Shape {shape} not supported")

    beta = w.sum(-1).ravel()
    X_ = rng.standard_normal((n_samples,) + shape)
    X = []
    for i in np.arange(n_samples):
        Xi = ndimage.gaussian_filter(X_[i], smooth_X)
        X.append(Xi.ravel())

    X = np.asarray(X)
    X_ = X.reshape((n_samples,) + shape)

    noise = sigma * rng.standard_normal(n_samples)
    y = np.dot(X, beta) + noise

    return X, y, beta, noise, X_, w


def multivariate_simulation_autoregressive(
    n_samples,
    n_features,
    n_times=None,
    support_size=10,
    rho=0.25,
    value=1.0,
    snr=2.0,
    rho_noise_time=0.0,
    shuffle=False,
    seed=None
):
    """
    Generate data with Toeplitz covariance structure and optional temporal correlation.

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features/variables 
    n_times : int or None, default=None
        Number of time points. None means single timepoint.
    support_size : int, default=10
        Number of non-zero coefficients
    rho : float, default=0.25
        Feature correlation coefficient
    value : float, default=1.0
        Value of non-zero coefficients
    snr : float, default=2.0
        Signal-to-noise ratio
    rho_noise_time : float, default=0.0
        Temporal noise correlation coefficient
    shuffle : bool, default=True
        Whether to shuffle features
    seed : int or None, default=None
        Random seed

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Design matrix
    y : ndarray, shape (n_samples,) or (n_samples, n_times)
        Target vector/matrix 
    beta_true : ndarray, shape (n_features,) or (n_features, n_times)
        True coefficients
    non_zero : ndarray
        Indices of non-zero coefficients
    noise_mag : float
        Noise magnitude scaling factor
    eps : ndarray, shape (n_samples,) or (n_samples, n_times)
        Noise vector/matrix
    """
    # Setup seed generator
    rng = np.random.default_rng(seed)

    # Generate the variables from a multivariate normal distribution
    mu = np.zeros(n_features)
    sigma = toeplitz(rho ** np.arange(0, n_features))  # covariance matrix of X
    X = rng.multivariate_normal(mu, sigma, size=(n_samples))

    # suffle the samples
    if shuffle:
        rng.shuffle(X.T)

    # Generate the response from a linear model
    non_zero = rng.choice(n_features, support_size, replace=False)
    if n_times is None:
        beta_true = np.zeros(n_features)
        beta_true[non_zero] = value
        eps = rng.standard_normal(size=n_samples)
    else:
        beta_true = np.zeros((n_features, n_times))
        beta_true[non_zero, :] = value
        # possibility to generate correlated noise
        sigma_time = toeplitz(rho_noise_time ** np.arange(0, n_times))  # covariance matrix of X
        eps = rng.multivariate_normal(np.zeros(n_times), sigma_time, size=(n_samples))
    prod_temp = np.dot(X, beta_true)
    if snr != 0.0:
        noise_mag = np.linalg.norm(prod_temp) / (snr * np.linalg.norm(eps))
    else:
        noise_mag = 0.0

    y = prod_temp + noise_mag * eps

    return X, y, beta_true, non_zero, noise_mag, eps
