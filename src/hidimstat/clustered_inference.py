import numpy as np
from sklearn.base import clone
from sklearn.utils import resample
from joblib import Parallel, delayed

from hidimstat.desparsified_lasso import (
    desparsified_lasso,
    desparsified_lasso_pvalue,
    desparsified_group_lasso_pvalue,
)
from hidimstat.stat_tools import aggregate_quantiles


def _subsampling(n_samples, train_size, groups=None, seed=0):
    """
    Random subsampling: computes a list of indices
    """
    index_row = np.arange(n_samples) if groups is None else np.unique(groups)
    train_index = resample(
        index_row,
        n_samples=int(len(index_row) * train_size),
        replace=False,
        random_state=seed,
    )
    if groups is not None:
        train_index = np.arange(n_samples)[np.isin(groups, train_index)]
    return train_index


def _degrouping(ward, beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr):
    """
    Assigning cluster-wise stats to features contained in the corresponding
    cluster and rescaling estimated parameter
    """
    # degroup variable other than beta_hat
    pval, pval_corr, one_minus_pval, one_minus_pval_corr = map(
        ward.inverse_transform,
        [pval, pval_corr, one_minus_pval, one_minus_pval_corr],
    )
    
    n_features = pval.shape[0]
    labels = ward.labels_
    # compute the size of each cluster
    clusters_size = np.zeros(labels.size)
    for label in range(labels.max() + 1):
        cluster_size = np.sum(labels == label)
        clusters_size[labels == label] = cluster_size
    # degroup beta_hat
    if len(beta_hat.shape) == 1:
        # weighting the weight of beta with the size of the cluster
        beta_hat = ward.inverse_transform(beta_hat) / clusters_size
    elif len(beta_hat.shape) == 2:
        n_times = beta_hat.shape[1]
        beta_hat_degrouped = np.zeros((n_features, n_times))
        for i in range(n_times):
            beta_hat_degrouped[:, i] = (
                ward.inverse_transform(beta_hat[:, i]) / clusters_size
            )
        beta_hat = beta_hat_degrouped


    return beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr


def clustered_inference(
    X_init,
    y,
    ward,
    n_clusters,
    scaler_sampling=None,
    train_size=1.0,
    groups=None,
    seed=0,
    n_jobs=1,
    verbose=1,
    **vi_kwargs,
):
    """Clustered inference algorithm

    Parameters
    ----------
    X_init : ndarray, shape (n_samples, n_features)
        Original data (uncompressed).

    y : ndarray, shape (n_samples,) or (n_samples, n_times)
        Target.

    ward : sklearn.cluster.FeatureAgglomeration
        Scikit-learn object that computes Ward hierarchical clustering.

    n_clusters : int
        Number of clusters used for the compression.

    train_size : float, optional (default=1.0)
        Fraction of samples used to compute the clustering.
        If `train_size = 1`, clustering is not random since all the samples
        are used to compute the clustering.

    groups : ndarray, shape (n_samples,), optional (default=None)
        Group labels for every sample. If not None, `groups` is used to build
        the subsamples that serve for computing the clustering.

    method : str, optional (default='desparsified-lasso')
        Method used for making the inference.
        Currently the two methods available are 'desparsified-lasso'
        and 'group-desparsified-lasso'. Use 'desparsified-lasso' for
        non-temporal data and 'group-desparsified-lasso' for temporal data.

    seed: int, optional (default=0)
        Seed used for generating a random subsample of the data.
        This seed controls the clustering randomness.

    n_jobs : int or None, optional (default=1)
        Number of CPUs to use during parallel steps such as inference.

    verbose: int, optional (default=1)
        The verbosity level. If `verbose > 0`, we print a message before
        runing the clustered inference.

    **kwargs:
        Arguments passed to the statistical inference function.

    Returns
    -------
    beta_hat : ndarray, shape (n_features,) or (n_features, n_times)
        Estimated parameter vector or matrix.

    pval : ndarray, shape (n_features,)
        p-value, with numerically accurate values for
        positive effects (ie., for p-value close to zero).

    pval_corr : ndarray, shape (n_features,)
        p-value corrected for multiple testing.

    one_minus_pval : ndarray, shape (n_features,)
        One minus the p-value, with numerically accurate values
        for negative effects (ie., for p-value close to one).

    one_minus_pval_corr : ndarray, shape (n_features,)
        One minus the p-value corrected for multiple testing.

    References
    ----------
    .. [1] Chevalier, J. A., Nguyen, T. B., Thirion, B., & Salmon, J. (2021).
           Spatially relaxed inference on high-dimensional linear models.
           arXiv preprint arXiv:2106.02590.
    """
    n_samples, n_features = X_init.shape

    if verbose > 0:
        print(
            f"Clustered inference: n_clusters = {n_clusters}, "
            + f"inference method desparsified lasso, seed = {seed}, groups = {groups is not None} "
        )

    ## This are the 3 step in first loop of the algorithm 2 of [1]
    # sampling row of X
    train_index = _subsampling(n_samples, train_size, groups=groups, seed=seed)

    # transformation matrix
    ward_ = clone(ward)
    ward_.fit(X_init[train_index, :])
    X_reduced = ward_.transform(X_init)

    # Preprocessing
    if scaler_sampling is not None:
        X_reduced = clone(scaler_sampling).fit_transform(X_reduced)

    # inference methods
    beta_hat, theta_hat, omega_diag = desparsified_lasso(
        X_reduced,
        y,
        group=len(y.shape)>1 and y.shape[1]>1, # detection of multiOutput
        n_jobs=n_jobs,
        verbose=verbose,
        **vi_kwargs,
    )

    return ward_, beta_hat, theta_hat, omega_diag


def clustered_inference_pvalue(
    n_samples, group, ward, beta_hat, theta_hat, omega_diag, **kwargs
):
    """
    family of corrected covariate-wise p-values
    """
    # corrected cluster-wise p-values
    if not group:
        pval, pval_corr, one_minus_pval, one_minus_pval_corr, cb_min, cb_max = (
            desparsified_lasso_pvalue(
                n_samples,
                beta_hat,
                theta_hat,
                omega_diag,
                **kwargs,
            )
        )
    else:
        pval, pval_corr, one_minus_pval, one_minus_pval_corr = (
            desparsified_group_lasso_pvalue(beta_hat, theta_hat, omega_diag, **kwargs)
        )

    # De-grouping
    beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = _degrouping(
        ward,
        beta_hat,
        pval,
        pval_corr,
        one_minus_pval,
        one_minus_pval_corr,
    )

    return beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr


def ensemble_clustered_inference(
    X_init,
    y,
    ward,
    n_clusters,
    scaler_sampling=None,
    train_size=0.3,
    groups=None,
    seed=0,
    n_bootstraps=25,
    n_jobs=None,
    verbose=1,
    **vi_kwargs,
):
    """Ensemble clustered inference algorithm
    Use desparsified lasso for inference methods

    Parameters
    ----------
    X_init : ndarray, shape (n_samples, n_features)
        Original data (uncompressed).

    y : ndarray, shape (n_samples,) or (n_samples, n_times)
        Target.

    ward : sklearn.cluster.FeatureAgglomeration
        Scikit-learn object that computes Ward hierarchical clustering.

    n_clusters : int
        Number of clusters used for the compression.

    train_size : float, optional (default=0.3)
        Fraction of samples used to compute the clustering.
        If `train_size = 1`, clustering is not random since all the samples
        are used to compute the clustering.

    groups : ndarray, shape (n_samples,), optional (default=None)
        Group labels for every sample. If not None, `groups` is used to build
        the subsamples that serve for computing the clustering.

    seed: int, optional (default=0)
        Seed used for generating a the first random subsample of the data.
        This seed controls the clustering randomness.

    n_bootstraps : int, optional (default=25)
        Number of clustered inference algorithm solutions to compute before
        making the ensembling.

    n_jobs : int or None, optional (default=1)
        Number of CPUs used to compute several clustered inference
        algorithms at the same time.

    verbose: int, optional (default=1)
        The verbosity level. If `verbose > 0`, we print a message before
        runing the clustered inference.

    **kwargs:
        Arguments passed to the statistical inference function.

    Returns
    -------
    beta_hat : ndarray, shape (n_features,) or (n_features, n_times)
        Estimated parameter vector or matrix.

    pval : ndarray, shape (n_features,)
        p-value, with numerically accurate values for
        positive effects (ie., for p-value close to zero).

    pval_corr : ndarray, shape (n_features,)
        p-value corrected for multiple testing.

    one_minus_pval : ndarray, shape (n_features,)
        One minus the p-value, with numerically accurate values
        for negative effects (ie., for p-value close to one).

    one_minus_pval_corr : ndarray, shape (n_features,)
        One minus the p-value corrected for multiple testing.

    References
    ----------
    .. [1] Chevalier, J. A., Nguyen, T. B., Thirion, B., & Salmon, J. (2021).
           Spatially relaxed inference on high-dimensional linear models.
           arXiv preprint arXiv:2106.02590.
    """
    # Clustered inference algorithms
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(clustered_inference)(
            X_init,
            y,
            ward,
            n_clusters,
            scaler_sampling=scaler_sampling,
            train_size=train_size,
            groups=groups,
            seed=i,
            n_jobs=1,
            verbose=verbose,
            **vi_kwargs,
        )
        for i in np.arange(seed, seed + n_bootstraps)
    )
    list_ward, list_beta_hat, list_theta_hat, list_omega_diag = [], [], [], []
    for ward, beta_hat, theta_hat, omega_diag in results:
        list_ward.append(ward)
        list_beta_hat.append(beta_hat)
        list_theta_hat.append(theta_hat)
        list_omega_diag.append(omega_diag)
    return list_ward, list_beta_hat, list_theta_hat, list_omega_diag


def ensemble_clustered_inference_pvalue(
    n_samples,
    group,
    list_ward,
    list_beta_hat,
    list_theta_hat,
    list_omega_diag,
    aggregate_method=aggregate_quantiles,
    n_jobs=None,
    verbose=0,
    **kwargs,
):
    """
    Aggregation result from bootstraping
    """
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(clustered_inference_pvalue)(
            n_samples,
            group,
            list_ward[i],
            list_beta_hat[i],
            list_theta_hat[i],
            list_omega_diag[i],
            **kwargs,
        )
        for i in range(len(list_ward))
    )
    # Collecting results
    list_beta_hat = []
    list_pval, list_pval_corr = [], []
    list_one_minus_pval, list_one_minus_pval_corr = [], []
    for beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr in results:
        list_beta_hat.append(beta_hat)
        list_pval.append(pval)
        list_pval_corr.append(pval_corr)
        list_one_minus_pval.append(one_minus_pval)
        list_one_minus_pval_corr.append(one_minus_pval_corr)

    # Ensembling
    beta_hat = np.mean(list_beta_hat, axis=0)
    results = [list_pval, list_pval_corr, list_one_minus_pval, list_one_minus_pval_corr]
    for index, data in enumerate(results):
        results[index] = aggregate_method(np.asarray(data))
    pval, pval_corr, one_minus_pval, one_minus_pval_corr = results

    return beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr
