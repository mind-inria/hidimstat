import numpy as np
from scipy import ndimage
from scipy.linalg import toeplitz


def multivariate_1D_simulation(
    n_samples=100,
    n_features=500,
    support_size=10,
    sigma=1.0,
    rho=0.0,
    shuffle=True,
    seed=0,
):
    """
    Generate 1D data with Toeplitz design matrix.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples.
    n_features : int, default=500
        Number of features.
    support_size : int, default=10
        Size of the support (number of non-zero coefficients).
    sigma : float, default=1.0
        Standard deviation of the additive White Gaussian noise.
    rho : float, default=0.0
        Level of correlation between neighboring features. Must be between 0 and 1.
    shuffle : bool, default=True
        If True, randomly shuffle the features to break 1D data structure.
    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Design matrix with Toeplitz correlation structure.
    y : ndarray of shape (n_samples,)
        Target vector y = X @ beta + noise.
    beta : ndarray of shape (n_features,)
        Parameter vector with support_size non-zero entries equal to 1.
    noise : ndarray of shape (n_samples,)
        Additive white Gaussian noise vector.
    """
    # Setup seed generator
    rng = np.random.default_rng(seed)

    # generate random data for each samples
    X = np.zeros((n_samples, n_features))
    X[:, 0] = rng.standard_normal(n_samples)
    for i in np.arange(1, n_features):
        rand_vector = ((1 - rho**2) ** 0.5) * rng.standard_normal(n_samples)
        X[:, i] = rho * X[:, i - 1] + rand_vector

    if shuffle:
        rng.shuffle(X.T)

    # generate the vector of variable of importances
    beta = np.zeros(n_features)
    beta[0:support_size] = 1.0

    # generate the simulated regression data
    noise = sigma * rng.standard_normal(n_samples)
    y = np.dot(X, beta) + noise

    return X, y, beta, noise


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


def multivariate_temporal_simulation(
    n_samples=100,
    n_features=500,
    n_times=30,
    support_size=10,
    sigma=1.0,
    rho_noise=0.0,
    rho_data=0.0,
    shuffle=True,
    seed=0,
):
    """
    Generate 1D temporal data with constant design matrix.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples.
    n_features : int, default=500
        Number of features.
    n_times : int, default=30
        Number of time points.
    support_size : int, default=10
        Size of the row support (number of non-zero coefficient rows).
    sigma : float, default=1.0
        Standard deviation of the additive white Gaussian noise.
    rho_noise : float, default=0.0
        Level of temporal autocorrelation in the noise. Must be between 0 and 1.
    rho_data : float, default=0.0
        Level of correlation between neighboring features. Must be between 0 and 1.
    shuffle : bool, default=True
        If True, randomly shuffle the features to break 1D data structure.
    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Design matrix with Toeplitz correlation structure.
    Y : ndarray of shape (n_samples, n_times)
        Target matrix Y = X @ beta + noise.
    beta : ndarray of shape (n_features, n_times)
        Parameter matrix with first support_size rows equal to 1.
    noise : ndarray of shape (n_samples, n_times)
        Temporally correlated Gaussian noise matrix.
    """

    rng = np.random.default_rng(seed)

    X = np.zeros((n_samples, n_features))
    X[:, 0] = rng.standard_normal(n_samples)

    for i in np.arange(1, n_features):
        rand_vector = ((1 - rho_data**2) ** 0.5) * rng.standard_normal(n_samples)
        X[:, i] = rho_data * X[:, i - 1] + rand_vector

    if shuffle:
        rng.shuffle(X.T)

    beta = np.zeros((n_features, n_times))
    beta[0:support_size, :] = 1.0

    noise = np.zeros((n_samples, n_times))
    noise[:, 0] = rng.standard_normal(n_samples)

    for i in range(1, n_times):
        rand_vector = ((1 - rho_noise**2) ** 0.5) * rng.standard_normal(n_samples)
        noise[:, i] = rho_noise * noise[:, i - 1] + rand_vector

    noise = sigma * noise

    Y = np.dot(X, beta) + noise

    return X, Y, beta, noise


def multivariate_1D_simulation_AR(
    n_samples, n_features, rho=0.25, snr=2.0, sparsity=0.06, sigma=1.0, seed=None
):
    """
    Function to simulate data follow an autoregressive structure with Toeplitz
    covariance matrix

    Parameters
    ----------
    n_samples : int
        number of observations
    n_features : int
        number of variables
    sparsity : float
        ratio of number of variables with non-zero coefficients over total
        coefficients
    rho : float
        Level of correlation between neighboring features.
    effect : float
        signal magnitude, value of non-null coefficients
    seed : None or Int
        random seed for generator

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Design matrix resulted from simulation
    y : ndarray, shape (n_samples, )
        Response vector resulted from simulation
    beta_true : ndarray, shape (n_samples, )
        Vector of true coefficient value
    non_zero : ndarray, shape (n_samples, )
        Vector of non zero coefficients index
    """
    # Setup seed generator
    rng = np.random.default_rng(seed)

    # Number of non-null
    k = int(sparsity * n_features)

    # Generate the variables from a multivariate normal distribution
    mu = np.zeros(n_features)
    Sigma = toeplitz(rho ** np.arange(0, n_features))  # covariance matrix of X
    X = rng.multivariate_normal(mu, Sigma, size=(n_samples))

    # Generate the response from a linear model
    non_zero = rng.choice(n_features, k, replace=False)
    beta_true = np.zeros(n_features)
    beta_true[non_zero] = sigma
    eps = rng.standard_normal(size=n_samples)
    prod_temp = np.dot(X, beta_true)
    noise_mag = np.linalg.norm(prod_temp) / (snr * np.linalg.norm(eps))
    y = prod_temp + noise_mag * eps

    return X, y, beta_true, non_zero
