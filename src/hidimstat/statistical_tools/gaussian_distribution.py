import warnings

import numpy as np
from sklearn.utils import check_random_state


class GaussianDistribution:
    """
    Generator for second-order Gaussian variables using the equi-correlated method.
    Creates synthetic variables that preserve the covariance structure of the original
    variables while ensuring conditional independence between the original and synthetic data.
    Parameters
    ----------
    cov_estimator : object
        Estimator for computing the covariance matrix. Must implement fit and
        have a covariance_ attribute after fitting.
    random_state : int or None, default=None
        Random seed for generating synthetic data.
    tol : float, default=1e-14
        Tolerance for numerical stability in matrix computations.
    Attributes
    ----------
    mu_tilde_ : ndarray of shape (n_samples, n_features)
        Mean matrix for generating synthetic variables.
    sigma_tilde_decompose_ : ndarray of shape (n_features, n_features)
        Cholesky decomposition of the synthetic covariance matrix.
    References
    ----------
    .. footbibliography::
    """

    def __init__(self, cov_estimator, random_state=None, tol=1e-14):
        self.cov_estimator = cov_estimator
        self.tol = tol
        self.rng = check_random_state(random_state)

    def fit(self, X):
        """
        Fit the Gaussian synthetic variable generator.
        This method estimates the parameters needed to generate Gaussian synthetic variables
        based on the input data. It follows a methodology for creating second-order
        synthetic variables that preserve the covariance structure.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples used to estimate the parameters for synthetic variable generation.
            The data is assumed to follow a Gaussian distribution.
        Returns
        -------
        self : object
            Returns the instance itself.
        Notes
        -----
        The method implements the following steps:
        1. Centers and scales the data if specified
        2. Estimates mean and covariance of input data
        3. Computes parameters for synthetic variable generation
        """
        _, n_features = X.shape

        # estimation of X distribution
        # original implementation:
        # https://github.com/msesia/knockoff-filter/blob/master/R/knockoff/R/create_second_order.R
        mu = X.mean(axis=0)
        sigma = self.cov_estimator.fit(X).covariance_

        diag_s = np.diag(_s_equi(sigma, tol=self.tol))

        sigma_inv_s = np.linalg.solve(sigma, diag_s)

        # First part on the RHS of equation 1.4 in barber2015controlling
        self.mu_tilde_ = X - np.dot(X - mu, sigma_inv_s)
        # To calculate the Cholesky decomposition later on
        sigma_tilde = 2 * diag_s - diag_s.dot(sigma_inv_s)
        # test is the matrix is positive definite
        while not np.all(np.linalg.eigvalsh(sigma_tilde) > self.tol):
            sigma_tilde += 1e-10 * np.eye(n_features)
            warnings.warn(
                "The conditional covariance matrix for knockoffs is not positive "
                "definite. Adding minor positive value to the matrix.",
                UserWarning,
            )

        self.sigma_tilde_decompose_ = np.linalg.cholesky(sigma_tilde)

        return self

    def _check_fit(self):
        """
        Check if the model has been fit before performing analysis.
        Raises
        ------
        ValueError
            If any of the required attributes are missing, indicating the model
            hasn't been fit before generating synthetic variables.
        """
        if not hasattr(self, "mu_tilde_") or not hasattr(
            self, "sigma_tilde_decompose_"
        ):
            raise ValueError("The GaussianGenerator requires to be fit before simulate")

    def sample(self):
        """
        Generate synthetic variables.
        This function generates synthetic variables that preserve the covariance structure
        of the original data while ensuring conditional independence.
        Returns
        -------
        X_tilde : 2D ndarray (n_samples, n_features)
            The synthetic variables.
        """
        self._check_fit()
        n_samples, n_features = self.mu_tilde_.shape

        # create a uniform noise for all the data
        u_tilde = self.rng.randn(n_samples, n_features)

        # Equation 1.4 in barber2015controlling
        X_tilde = self.mu_tilde_ + np.dot(u_tilde, self.sigma_tilde_decompose_)
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
