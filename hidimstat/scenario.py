import numpy as np
from scipy import ndimage

ROI_SIZE_2D = 2
SHAPE_2D = (12, 12)

ROI_SIZE_3D = 2
SHAPE_3D = (12, 12, 12)


def multivariate_1D_simulation(
    n_samples=100,
    n_features=500,
    support_size=10,
    sigma=1.0,
    rho=0.0,
    shuffle=True,
    seed=0,
    nb_group=0,
    group_size=0,
):
    """Generate 1D data with Toeplitz design matrix

    Parameters
    -----------
    n_samples : int
        Number of samples.

    n_features : int
        Number of features.

    support_size : int
        Size of the support.

    sigma : float
        Standard deviation of the additive White Gaussian noise.

    rho: float
        Level of correlation between neighboring features (if not `shuffle`).

    shuffle : bool
        Shuffle the features (breaking 1D data structure) if True.

    seed : int
        Seed used for generating design matrix and noise.

    nb_group : int
        Number of groups.

    group_size : int
        Size of each group.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Design matrix.
        if their is some groups, the first rows contains the groups values.

    y : ndarray, shape (n_samples,)
        Target.

    beta : ndarray, shape (n_features,)
        Parameter vector.

    noise : ndarray, shape (n_samples,)
        Additive white Gaussian noise.
    """

    rng = np.random.default_rng(seed)
    if nb_group < 0 or group_size < 0:
        raise ValueError("The number of groups and their size must be positive.")
    n_generate_samples = n_samples - nb_group * group_size
    if n_generate_samples <= 0:
        raise ValueError(
            "The number of samples is too small compate to the number "
            "of group and their size to gerate the data."
        )
    X = np.zeros((n_generate_samples, n_features))
    X[:, 0] = rng.standard_normal(n_generate_samples)

    for i in np.arange(1, n_features):
        rand_vector = ((1 - rho**2) ** 0.5) * rng.standard_normal(n_generate_samples)
        X[:, i] = rho * X[:, i - 1] + rand_vector

    if shuffle:
        rng.shuffle(X.T)

    if nb_group > 0:
        groups = np.repeat(X[:nb_group], group_size, axis=0)
        X = np.vstack((groups, X))

    beta = np.zeros(n_features)
    beta[0:support_size] = 1.0

    noise = sigma * rng.standard_normal(n_samples)
    y = np.dot(X, beta) + noise

    return X, y, beta, noise


def generate_2D_weight(shape, roi_size):
    """Create a 2D weight map with four ROIs

    Parameters
    -----------
    shape : tuple (n_x, n_z)
        Shape of the data in the simulation.

    roi_size : int
        Size of the edge of the ROIs.

    Returns
    -------
    w : ndarray, shape (n_x, n_z)
        2D weight map.
    """

    w = np.zeros(shape + (5,))
    w[0:roi_size, 0:roi_size, 0] = 1.0
    w[-roi_size:, -roi_size:, 1] = 1.0
    w[0:roi_size, -roi_size:, 2] = 1.0
    w[-roi_size:, 0:roi_size, 3] = 1.0

    return w


def generate_3D_weight(shape, roi_size):
    """Create a 3D weight map with five ROIs

    Parameters
    -----------
    shape : tuple (n_x, n_y, n_z)
        Shape of the data in the simulation.

    roi_size : int
        Size of the edge of the ROIs.

    Returns
    -------
    w : ndarray, shape (n_x, n_y, n_z)
        3D weight map.
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
    shape=SHAPE_2D,
    roi_size=ROI_SIZE_2D,
    sigma=1.0,
    smooth_X=1.0,
    return_shaped_data=True,
    seed=0,
):
    """Generate a multivariate simulation with 2D or 3D data

    Parameters
    -----------
    n_samples : int
        Number of samples.

    shape : tuple (n_x, n_y) or (n_x, n_y, n_z)
        Shape of the data in the simulation.

    roi_size : int
        Size of the edge of the ROIs.

    sigma : float
        Standard deviation of the additive white Gaussian noise.

    smooth_X : float
        Level of (data) smoothing using a Gaussian filter.

    return_shaped_data : bool
        If true, the function returns shaped data and weight map.

    seed : int
        Seed used for generating design matrix and noise.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Design matrix.

    y : ndarray, shape (n_samples,)
        Target.
    beta: ndarray, shape (n_features,)
        Parameter vector (flattened weight map).

    noise: ndarray, shape (n_samples,)
        Additive white Gaussian noise.

    X_: ndarray, shape (n_samples, n_x, n_y) or (n_samples, n_x, n_y, n_z)
        Reshaped design matrix.

    w : ndarray, shape (n_x, n_y) or (n_x, n_y, n_z)
        2D or 3D weight map.
    """

    rng = np.random.default_rng(seed)

    if len(shape) == 2:
        w = generate_2D_weight(shape, roi_size)
    elif len(shape) == 3:
        w = generate_3D_weight(shape, roi_size)

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

    if return_shaped_data:
        return X, y, beta, noise, X_, w

    return X, y, beta, noise


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
    """Generate 1D temporal data with constant design matrix

    Parameters
    -----------
    n_samples : int
        Number of samples.

    n_features : int
        Number of features.

    n_times : int
        Number of time points.

    support_size: int
        Size of the row support.

    sigma : float
        Standard deviation of the noise at each time point.

    rho_noise : float
        Level of autocorrelation in the noise.

    rho_data: float
        Level of correlation between neighboring features (if not `shuffle`).

    shuffle : bool
        Shuffle the features (breaking 1D data structure) if True.

    seed : int
        Seed used for generating design matrix and noise.

    Returns
    -------
    X: ndarray, shape (n_samples, n_features)
        Design matrix.

    Y : ndarray, shape (n_samples, n_times)
        Target.

    beta : ndarray, shape (n_features, n_times)
        Parameter matrix.

    noise : ndarray, shape (n_samples, n_times)
        Noise matrix.
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
