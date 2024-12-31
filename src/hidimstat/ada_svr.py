import numpy as np
from hidimstat.stat_tools import pval_from_scale

__all__ = ["ada_svr", "ada_svr_pvalue"]


def ada_svr(X, y, rcond=1e-3):
    """
    ADA-SVR: Adaptive Permutation Threshold Support Vector Regression

    Statistical inference procedure presented in :footcite:t:`gaonkar_deriving_2012`.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data.

    y : ndarray, shape (n_samples,)
        Target.

    rcond : float, optional (default=1e-3)
        Cutoff for small singular values. Singular values smaller
        than `rcond` * largest_singular_value are set to zero.
        Deafult value is 1e-3.

    Returns
    -------
    beta_hat : array, shape (n_features,)
        Estimated parameter vector.

    scale : ndarray, shape (n_features,)
        Value of the standard deviation of the parameters.

    References
    ----------
    .. footbibliography::

    """
    X_input = np.asarray(X)

    ## compute matrix C, (see eq.6 of [1])
    # invert matrix X*X'
    XXT_inv = np.linalg.pinv(np.dot(X_input, X_input.T), rcond=rcond)
    # partial computation of the 2nd term of the equation
    sum_XXT_inv = np.sum(XXT_inv)
    L = -np.outer(np.sum(XXT_inv, axis=0), np.sum(XXT_inv, axis=1)) / sum_XXT_inv
    C = np.dot(X_input.T, XXT_inv + L)

    ## compute vector W, (see eq.4 of [1])
    beta_hat = np.dot(C, y)
    ## compute standard deviation of the distribution of W, (see eq.12 of [1])
    scale = np.sqrt(np.sum(C**2, axis=1))

    return beta_hat, scale


def ada_svr_pvalue(beta_hat, scale, distrib="norm", eps=1e-14):
    """
    Computing one-sided p-values corrrected for multiple testing
    from simple testing one-sided p-values.

    For details see: :py:func:`hidimstat.pval_from_scale`

    """
    return pval_from_scale(beta_hat, scale, distrib=distrib, eps=eps)
