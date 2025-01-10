import warnings
import numpy as np


def gaussian_knockoff_generation(X, mu, sigma, seed=None, tol=1e-14, repeat=False):
    """
    Generate second-order knockoff variables using equi-correlated method.
    Reference: Candes et al. (2016), Barber et al. (2015)
    
    original impelmentation:
    https://github.com/msesia/knockoff-filter/blob/master/R/knockoff/R/create_gaussian.R
    
    Parameters
    ----------
    X: 2D ndarray (n_samples, n_features)
        original design matrix

    mu : 1D ndarray (n_features, )
        vector of empirical mean values

    method: str
        method to generate gaussian knockoff

    sigma : 2D ndarray (n_samples, n_features)
        empirical covariance matrix

    Returns
    -------
    X_tilde : 2D ndarray (n_samples, n_features)
        knockoff design matrix
    """
    n_samples, n_features = X.shape

    # create a uniform noise for all the data
    rng = np.random.RandomState(seed)
    U_tilde = rng.randn(n_samples, n_features)

    Diag_s = np.diag(_s_equi(sigma, tol=tol))

    Sigma_inv_s = np.linalg.solve(sigma, Diag_s)

    # First part on the RHS of equation 1.4 in Barber & Candes (2015)
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

    # Equation 1.4 in Barber & Candes (2015)
    sigma_tilde_decompose = np.linalg.cholesky(sigma_tilde)
    X_tilde = Mu_tilde + np.dot(U_tilde, sigma_tilde_decompose)

    if not repeat:
        return X_tilde
    else:
        return X_tilde, (Mu_tilde, sigma_tilde_decompose)


def repeat_gaussian_knockoff_generation(Mu_tilde, sigma_tilde_decompose, seed):
    n_samples, n_features = Mu_tilde.shape/2
    # create a uniform noise for all the data
    rng = np.random.RandomState(seed)
    U_tilde = rng.randn(n_samples, n_features)

    X_tilde = Mu_tilde + np.dot(U_tilde, sigma_tilde_decompose)
    return X_tilde
     


def _s_equi(sigma, tol=1e-14):
    """Estimate diagonal matrix of correlation between real and knockoff
    variables using equi-correlated equation
    
    original implementation: 
    https://github.com/msesia/knockoff-filter/blob/master/R/knockoff/R/solve_equi.R

    Parameters
    ----------
    sigma : 2D ndarray (n_features, n_features)
        empirical covariance matrix calculated from original design matrix

    Returns
    -------
    1D ndarray (n_features, )
        vector of diagonal values of estimated matrix diag{s}
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
        else:
            s_eps *= 10
        # if all eigval > 0 then the matrix is positive define
        psd = np.all(np.linalg.eigvalsh(2 * corr_matrix - np.diag(S * (1 - s_eps))) > tol)


    S = S * (1 - s_eps)

    return S * np.diag(sigma)
