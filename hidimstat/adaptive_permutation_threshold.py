import numpy as np


def ada_svr(X, y, rcond=1e-3):
    """Statistical inference procedure presented in Gaonkar et al. [1]_.

    Parameters
    -----------
    X : ndarray, shape (n_samples, n_features)
        Data.

    y : ndarray, shape (n_samples,)
        Target.

    rcond : float, optional (default=1e-3)
        Cutoff for small singular values. Singular values smaller
        than `rcond` * largest_singular_value are set to zero.

    Returns
    -------
    beta_hat : array, shape (n_features,)
        Estimated parameter vector.

    scale : ndarray, shape (n_features,)
        Value of the standard deviation of the parameters.

    References
    ----------
    .. [1] Gaonkar, B., & Davatzikos, C. (2012, October). Deriving statistical
           significance maps for SVM based image classification and group
           comparisons. In International Conference on Medical Image Computing
           and Computer-Assisted Intervention (pp. 723-730). Springer, Berlin,
           Heidelberg.
    """

    X = np.asarray(X)
    n_samples, n_features = X.shape

    K = _manual_inversion(np.dot(X, X.T), rcond=rcond)
    sum_K = np.sum(K)

    L = -np.outer(np.sum(K, axis=0), np.sum(K, axis=1)) / sum_K
    C = np.dot(X.T, K + L)

    beta_hat = np.dot(C, y)

    scale = np.sqrt(np.sum(C**2, axis=1))

    return beta_hat, scale


def _manual_inversion(X, rcond=1e-3):
    """
    Inverting taking care of low eigenvalues to increase numerical stability
    
    Parameters
    -----------
    X : ndarray, shape (n_features, n_features) 
        Data.
        Base on usage of X in the main function, X is squared matrix and not full_ranked (for details see PR#58)
        
    """

    X = np.asarray(X)
    U, s, V = np.linalg.svd(X, full_matrices=False)
    rank = np.sum(s > rcond * s.max())
    s_inv = np.zeros(np.size(s))
    s_inv[:rank] = 1 / s[:rank]

    X_inv = np.linalg.multi_dot([U, np.diag(s_inv), V])

    return X_inv
