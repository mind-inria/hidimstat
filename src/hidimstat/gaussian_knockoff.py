import warnings

import numpy as np

from hidimstat._utils.utils import check_random_state


def gaussian_knockoff_generation(X, mu, sigma, random_state=None, tol=1e-14):
    """
    Generate second-order knockoff variables using the equi-correlated method.

    This function generates knockoff variables for a given design matrix X,
    using the equi-correlated method described in :footcite:t:`barber2015controlling`
    and :footcite:t:`candes2018panning`. The function takes as input the design matrix
    X, the vector of empirical mean values mu, and the empirical covariance
    matrix sigma. It returns the knockoff variables X_tilde.

    The original implementation can be found at
    https://github.com/msesia/knockoff-filter/blob/master/R/knockoff/R/create_gaussian.R

    Parameters
    ----------
    X: 2D ndarray (n_samples, n_features)
        The original design matrix.
    mu : 1D ndarray (n_features, )
        A vector of empirical mean values.
    sigma : 2D ndarray (n_samples, n_features)
        The empirical covariance matrix.
    random_state : int or None, default=None
        A random seed for generating the uniform noise used to create
        the knockoff variables.
    tol : float, default=1.e-14
        A tolerance value used for numerical stability in the calculation
        of the Cholesky decomposition.

    Returns
    -------
    X_tilde : 2D ndarray (n_samples, n_features)
        The knockoff variables.
    mu_tilde : 2D ndarray (n_samples, n_features)
        The mean matrix used for generating knockoffs.
    sigma_tilde_decompose : 2D ndarray (n_features, n_features)
        The Cholesky decomposition of the covariance matrix.

    References
    ----------
    .. footbibliography::
    """
    n_samples, n_features = X.shape

    # create a uniform noise for all the data
    rng = check_random_state(random_state)
    u_tilde = rng.standard_normal((n_samples, n_features))

    diag_s = np.diag(_s_equi(sigma, tol=tol))

    sigma_inv_s = np.linalg.solve(sigma, diag_s)

    # First part on the RHS of equation 1.4 in barber2015controlling
    mu_tilde = X - np.dot(X - mu, sigma_inv_s)
    # To calculate the Cholesky decomposition later on
    sigma_tilde = 2 * diag_s - diag_s.dot(sigma_inv_s)
    # test is the matrix is positive definite
    while not np.all(np.linalg.eigvalsh(sigma_tilde) > tol):
        sigma_tilde += 1e-10 * np.eye(n_features)
        warnings.warn(
            "The conditional covariance matrix for knockoffs is not positive "
            "definite. Adding minor positive value to the matrix.",
            UserWarning,
        )

    # Equation 1.4 in barber2015controlling
    sigma_tilde_decompose = np.linalg.cholesky(sigma_tilde)
    X_tilde = mu_tilde + np.dot(u_tilde, sigma_tilde_decompose)

    return X_tilde, mu_tilde, sigma_tilde_decompose


def repeat_gaussian_knockoff_generation(mu_tilde, sigma_tilde_decompose, random_state):
    """
    Generate additional knockoff variables using pre-computed values.

    This function generates additional knockoff variables using pre-computed
    values returned by the gaussian_knockoff_generation function
    with repeat=True. It takes as input mu_tilde and sigma_tilde_decompose,
    which were returned by gaussian_knockoff_generation, and a random seed.
    It returns the new knockoff variables X_tilde.

    Parameters
    ----------
    mu_tilde : 2D ndarray (n_samples, n_features)
        The matrix of means used to generate the knockoff variables,
        returned by gaussian_knockoff_generation.

    sigma_tilde_decompose : 2D ndarray (n_features, n_features)
        The Cholesky decomposition of the covariance matrix used
        to generate the knockoff variables,returned by
        gaussian_knockoff_generation.

    random_state : int
        A random seed for generating the uniform noise used to create
        the knockoff variables.

    Returns
    -------
    X_tilde : 2D ndarray (n_samples, n_features)
        The knockoff variables.
    """
    n_samples, n_features = mu_tilde.shape

    # create a uniform noise for all the data
    rng = check_random_state(random_state)
    u_tilde = rng.standard_normal((n_samples, n_features))

    X_tilde = mu_tilde + np.dot(u_tilde, sigma_tilde_decompose)
    return X_tilde


def _s_equi(sigma, tol=1e-14):
    """
    Estimate the diagonal matrix of correlation between real
    and knockoff variables using the equi-correlated equation.

    This function estimates the diagonal matrix of correlation
    between real and knockoff variables using the equi-correlated
    equation described in :footcite:t:`barber2015controlling` and
    :footcite:t:`candes2018panning`. It takes as input the empirical
    covariance matrix sigma and a tolerance value tol,
    and returns a vector of diagonal values of the estimated
    matrix diag{s}.

    Parameters
    ----------
    sigma : 2D ndarray (n_features, n_features)
        The empirical covariance matrix calculated from
        the original design matrix.

    tol : float, optional
        A tolerance value used for numerical stability in the calculation
        of the eigenvalues of the correlation matrix.

    Returns
    -------
    1D ndarray (n_features, )
        A vector of diagonal values of the estimated matrix diag{s}.

    Raises
    ------
    Exception
        If the covariance matrix is not positive-definite.
    """
    n_features = sigma.shape[0]

    # Convert covariance matrix to correlation matrix
    # as example see cov2corr from statsmodels
    features_std = np.sqrt(np.diag(sigma))
    scale = np.outer(features_std, features_std)
    corr_matrix = sigma / scale

    eig_value = np.linalg.eigvalsh(corr_matrix)
    lambda_min = np.min(eig_value[0])
    # check if the matrix is positive-defined
    if lambda_min <= 0:
        raise Exception("The covariance matrix is not positive-definite.")

    s = np.ones(n_features) * min(2 * lambda_min, 1)

    psd = np.all(np.linalg.eigvalsh(2 * corr_matrix - np.diag(s)) > tol)
    s_eps = 0
    while not psd:
        if s_eps == 0:
            s_eps = np.finfo(type(s[0])).eps  # avoid zeros
        else:
            s_eps *= 10
        # if all eigval > 0 then the matrix is positive define
        psd = np.all(
            np.linalg.eigvalsh(2 * corr_matrix - np.diag(s * (1 - s_eps))) > tol
        )
        warnings.warn(
            "The equi-correlated matrix for knockoffs is not positive "
            f"definite. Reduce the value of distance by {s_eps}.",
            UserWarning,
        )

    s = s * (1 - s_eps)

    return s * np.diag(sigma)
