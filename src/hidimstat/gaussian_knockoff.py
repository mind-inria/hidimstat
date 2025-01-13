import warnings
import numpy as np


def gaussian_knockoff_generation(X, mu, sigma, seed=None, tol=1e-14, repeat=False):
    """
    Generate second-order knockoff variables using the equi-correlated method.

    This function generates knockoff variables for a given design matrix X,
    using the equi-correlated method described in :cite:`barber2015controlling` and
    :cite:`candes2018panning`. The function takes as input the design matrix X,
    the vector of empirical mean values mu, and the empirical covariance
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

    seed : int, optional
        A random seed for generating the uniform noise used to create the knockoff variables.

    tol : float, optional
        A tolerance value used for numerical stability in the calculation of the Cholesky decomposition.

    repeat : bool, optional
        If True, the function returns the values used to generate the knockoff variables (Mu_tilde and sigma_tilde_decompose),
        which can be used to generate additional knockoff variables without having to recompute them.

    Returns
    -------
    X_tilde : 2D ndarray (n_samples, n_features)
        The knockoff variables.

    References
    ----------
    .. footbibliography::
    """
    n_samples, n_features = X.shape

    # create a uniform noise for all the data
    rng = np.random.RandomState(seed)
    U_tilde = rng.randn(n_samples, n_features)

    Diag_s = np.diag(_s_equi(sigma, tol=tol))

    Sigma_inv_s = np.linalg.solve(sigma, Diag_s)

    # First part on the RHS of equation 1.4 in barber2015controlling
    Mu_tilde = X - np.dot(X - mu, Sigma_inv_s)
    # To calculate the Cholesky decomposition later on
    sigma_tilde = 2 * Diag_s - Diag_s.dot(Sigma_inv_s.dot(Diag_s)) #TODO extra operation Sigma_inv_s.dot(Diag_s) ??????
    # test is the matrix is positive definite
    while not np.all(np.linalg.eigvalsh(sigma_tilde) > tol): 
        sigma_tilde += 1e-10 * np.eye(n_features)
        warnings.warn(
            "The conditional covariance matrix for knockoffs is not positive "
            "definite. Adding minor positive value to the matrix.", UserWarning
        )

    # Equation 1.4 in barber2015controlling
    sigma_tilde_decompose = np.linalg.cholesky(sigma_tilde)
    X_tilde = Mu_tilde + np.dot(U_tilde, sigma_tilde_decompose)

    if not repeat:
        return X_tilde
    else:
        return X_tilde, (Mu_tilde, sigma_tilde_decompose)


def repeat_gaussian_knockoff_generation(Mu_tilde, sigma_tilde_decompose, seed):
    """
    Generate additional knockoff variables using pre-computed values.

    This function generates additional knockoff variables using pre-computed values returned by the
    gaussian_knockoff_generation function with repeat=True. It takes as input Mu_tilde and
    sigma_tilde_decompose, which were returned by gaussian_knockoff_generation, and a random seed.
    It returns the new knockoff variables X_tilde.

    Parameters
    ----------
    Mu_tilde : 2D ndarray (n_samples, n_features)
        The matrix of means used to generate the knockoff variables, returned by gaussian_knockoff_generation.

    sigma_tilde_decompose : 2D ndarray (n_features, n_features)
        The Cholesky decomposition of the covariance matrix used to generate the knockoff variables,
        returned by gaussian_knockoff_generation.

    seed : int
        A random seed for generating the uniform noise used to create the knockoff variables.

    Returns
    -------
    X_tilde : 2D ndarray (n_samples, n_features)
        The knockoff variables.
    """
    n_samples, n_features = Mu_tilde.shape
    
    # create a uniform noise for all the data
    rng = np.random.RandomState(seed)
    U_tilde = rng.randn(n_samples, n_features)

    X_tilde = Mu_tilde + np.dot(U_tilde, sigma_tilde_decompose)
    return X_tilde
     


def _s_equi(sigma, tol=1e-14):
    """
    Estimate the diagonal matrix of correlation between real and knockoff variables using the equi-correlated equation.

    This function estimates the diagonal matrix of correlation between real and knockoff variables using the
    equi-correlated equation described in :cite:`barber2015controlling` and
    :cite:`candes2018panning`. It takes as input the empirical covariance matrix sigma and a tolerance value tol, 
    and returns a vector of diagonal values of the estimated matrix diag{s}.

    Parameters
    ----------
    sigma : 2D ndarray (n_features, n_features)
        The empirical covariance matrix calculated from the original design matrix.

    tol : float, optional
        A tolerance value used for numerical stability in the calculation of the eigenvalues of the correlation matrix.

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
    if lambda_min < 0:
        raise Exception('The covariance matrix is not positive-definite.')

    S = np.ones(n_features) * min(2 * lambda_min, 1)
    
    psd = np.all(np.linalg.eigvalsh(2 * corr_matrix - np.diag(S)) > tol)
    s_eps = 0
    while psd is False:
        if s_eps == 0:
            s_eps = np.eps  # avoid zeros
            s_eps = np.eps
            s_eps *= 10
        # if all eigval > 0 then the matrix is positive define
        psd = np.all(np.linalg.eigvalsh(2 * corr_matrix - np.diag(S * (1 - s_eps))) > tol)

    S = S * (1 - s_eps)

    return S * np.diag(sigma)

