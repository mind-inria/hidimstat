import numpy as np
import pandas as pd
from hidimstat.stat_tools import pval_from_scale
from typing import Any


def ada_svr(
    X: np.ndarray[Any, np.dtype[np.float64]],
    y: np.ndarray[Any, np.dtype[np.float64]],
    rcond: float = 1e-3,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]
]:
    """
    ADA-SVR: Adaptive Permutation Threshold Support Vector Regression

    | Statistical inference procedure presented in :footcite:t:`gaonkar_deriving_2012`.
    | For example of usage see :ref:`example <sphx_glr_auto_examples_methods_ada_svr.py>`.

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
    "X must be a numpy array or a list or a pandas DataFrame"
    assert (isinstance(X, np.ndarray) or isinstance(X, list)) or isinstance(X, pd.DataFrame)
    "y must be a numpy array or a list or a pandas DataFrame"
    assert (isinstance(y, np.ndarray) or isinstance(y, list)) or isinstance(y, pd.DataFrame)
    assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
    assert X.ndim == 2, "X must be a 2D array"
    assert y.ndim == 1, "y must be a 1D array"
    assert X.shape[0] > 1, "X must have at least 2 samples"
    assert X.shape[1] > 0, "X must have at least 1 feature"
    assert isinstance(rcond, float), "rcond must be a float"
    assert rcond > 0, "rcond must be positive"

    X_input: np.ndarray[Any, np.dtype[np.float64]] = np.asarray(X)

    ## compute matrix C, (see eq.6 of [1])
    # invert matrix X*X'
    XXT_inv: np.ndarray[Any, np.dtype[np.float64]] = np.linalg.pinv(
        np.dot(X_input, X_input.T), rcond=rcond
    )
    # partial computation of the 2nd term of the equation
    sum_XXT_inv: np.ndarray[Any, np.dtype[np.float64]] = np.sum(XXT_inv)
    L: np.ndarray[Any, np.dtype[np.float64]] = (
        -np.outer(np.sum(XXT_inv, axis=0), np.sum(XXT_inv, axis=1)) / sum_XXT_inv
    )
    C: np.ndarray[Any, np.dtype[np.float64]] = np.dot(X_input.T, XXT_inv + L)

    ## compute vector W, (see eq.4 of [1])
    beta_hat: np.ndarray[Any, np.dtype[np.float64]] = np.dot(C, y)
    ## compute standard deviation of the distribution of W, (see eq.12 of [1])
    scale: np.ndarray[Any, np.dtype[np.float64]] = np.sqrt(np.sum(C**2, axis=1))

    assert beta_hat.shape[0] == X.shape[1], "beta_hat must have the same number of features as X"
    assert beta_hat.ndim == 1, "beta_hat must be a 1D array"
    assert np.all(np.isfinite(beta_hat)), "beta_hat must be finite values"
    assert scale.shape[0] == X.shape[1], "scale must have the same number of features as X"
    assert scale.ndim == 1, "scale must be a 1D array"
    assert np.all(scale >= 0), "scale must be positive"
    assert np.all(np.isfinite(scale)), "scale must be finite values"

    return beta_hat, scale


def ada_svr_pvalue(
    beta_hat: np.ndarray[Any, np.dtype[np.float64]],
    scale: np.ndarray[Any, np.dtype[np.float64]],
    distrib: str = "norm",
    eps: float = 1e-14,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    """
    Computing one-sided p-values corrrected for multiple testing
    from simple testing one-sided p-values.

    For details see: :py:func:`hidimstat.pval_from_scale`

    """
    return pval_from_scale(beta_hat, scale, distrib=distrib, eps=eps)
