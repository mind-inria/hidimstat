# -*- coding: utf-8 -*-
# Authors: Binh Nguyen <tuan-binh.nguyen@inria.fr>
"""
Implementation of Model-X knockoffs inference procedure, introduced in
Candes et. al. (2016) " Panning for Gold: Model-X Knockoffs for
High-dimensional Controlled Variable Selection"
<https://arxiv.org/abs/1610.02351>
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_memory

from .gaussian_knockoff import _estimate_distribution, gaussian_knockoff_generation
from .stat_coef_diff import _coef_diff_threshold, stat_coef_diff


def model_x_knockoff(
    X,
    y,
    fdr=0.1,
    offset=1,
    method="equi",
    statistics="lasso_cv",
    shrink=False,
    centered=True,
    cov_estimator="ledoit_wolf",
    verbose=False,
    memory=None,
    n_jobs=1,
    seed=None,
):
    """Model-X Knockoff

    It's an inference procedure to control False Discoveries Rate,
    based on :footcite:t:`candesPanningGoldModelX2017`

    Parameters
    ----------
    X : 2D ndarray (n_samples, n_features)
        design matrix

    y : 1D ndarray (n_samples, )
        response vector

    fdr : float, optional
        desired controlled FDR level

    offset : int, 0 or 1, optional
        offset to calculate knockoff threshold, offset = 1 is equivalent to
        knockoff+

    method : str, optional
        knockoff construction methods, either equi for equi-correlated knockoff
        or sdp for optimization scheme

    statistics : str, optional
        method to calculate knockoff test score

    shrink : bool, optional
        whether to shrink the empirical covariance matrix

    centered : bool, optional
        whether to standardize the data before doing the inference procedure

    cov_estimator : str, optional
        method of empirical covariance matrix estimation

    seed : int or None, optional
        random seed used to generate Gaussian knockoff variable

    Returns
    -------
    selected : 1D array, int
        vector of index of selected variables

    test_score : 1D array, (n_features, )
        vector of test statistic

    thres : float
        knockoff threshold

    X_tilde : 2D array, (n_samples, n_features)
        knockoff design matrix

    References
    ----------
    .. footbibliography::
    """
    memory = check_memory(memory)

    if centered:
        X = StandardScaler().fit_transform(X)

    mu, Sigma = _estimate_distribution(X, shrink=shrink, cov_estimator=cov_estimator)

    X_tilde = gaussian_knockoff_generation(
        X, mu, Sigma, memory=memory, method=method, seed=seed
    )
    test_score = memory.cache(stat_coef_diff, ignore=["n_jobs", "joblib_verbose"])(
        X, X_tilde, y, method=statistics, n_jobs=n_jobs
    )
    thres = _coef_diff_threshold(test_score, fdr=fdr, offset=offset)

    selected = np.where(test_score >= thres)[0]

    if verbose:
        return selected, test_score, thres, X_tilde

    return selected
