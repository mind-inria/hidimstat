import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.utils.validation import check_memory

from .desparsified_lasso import (
    desparsified_lasso,
    desparsified_lasso_pvalue,
    desparsified_group_lasso_pvalue,
)


def _subsampling(n_samples, train_size, groups=None, seed=0):
    """Random subsampling: computes a list of indices"""

    if groups is None:

        n_subsamples = int(n_samples * train_size)
        train_index = resample(
            np.arange(n_samples),
            n_samples=n_subsamples,
            replace=False,
            random_state=seed,
        )

    else:

        unique_groups = np.unique(groups)
        n_groups = unique_groups.size
        n_subsample_groups = int(n_groups * train_size)
        train_group = resample(
            unique_groups,
            n_samples=n_subsample_groups,
            replace=False,
            random_state=seed,
        )
        train_index = np.arange(n_samples)[np.isin(groups, train_group)]

    return train_index


def _ward_clustering(X_init, ward, train_index):
    """Ward clustering applied to full X but computed from a subsample of X"""

    ward = ward.fit(X_init[train_index, :])
    X_reduced = ward.transform(X_init)

    return X_reduced, ward


def hd_inference(X, y, method, n_jobs=1, memory=None, verbose=0, **kwargs):
    """Wrap-up high-dimensional inference procedures

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data.

    y : ndarray, shape (n_samples,) or (n_samples, n_times)
        Target.

    method : str, optional (default='desparsified-lasso')
        Method used for making the inference.
        Currently the two methods available are 'desparsified-lasso'
        and 'group-desparsified-lasso'. Use 'desparsified-lasso' for
        non-temporal data and 'group-desparsified-lasso' for temporal data.

    n_jobs : int or None, optional (default=1)
        Number of CPUs to use during parallel steps such as inference.

    memory : str or joblib.Memory object, optional (default=None)
        Used to cache the output of the computation of the clustering
        and the inference. By default, no caching is done. If a string is
        given, it is the path to the caching directory.

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
    """
    if method not in ["desparsified-lasso", "desparsified-group-lasso"]:
        raise ValueError("Unknow method")
    group = method == "desparsified-group-lasso"
    print("hd_inference", group, kwargs)
    beta_hat, theta_hat, precision_diag = desparsified_lasso(
        X, y, group=group, n_jobs=n_jobs, verbose=verbose, memory=memory, **kwargs
    )
    if not group:
        (
            pval,
            pval_corr,
            one_minus_pval,
            one_minus_pval_corr,
            cb_min,
            cb_max,
        ) = desparsified_lasso_pvalue(
            X.shape[0],
            beta_hat,
            theta_hat,
            precision_diag,
            confidence=0.95,
            **kwargs,
        )
    else:
        pval, pval_corr, one_minus_pval, one_minus_pval_corr = (
            desparsified_group_lasso_pvalue(
                beta_hat, theta_hat, precision_diag, **kwargs
            )
        )
    return beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr


def _degrouping(ward, beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr):
    """Assigning cluster-wise stats to features contained in the corresponding
    cluster and rescaling estimated parameter"""

    pval_degrouped = ward.inverse_transform(pval)
    pval_corr_degrouped = ward.inverse_transform(pval_corr)
    one_minus_pval_degrouped = ward.inverse_transform(one_minus_pval)
    one_minus_pval_corr_degrouped = ward.inverse_transform(one_minus_pval_corr)

    labels = ward.labels_
    clusters_size = np.zeros(labels.size)

    for label in range(labels.max() + 1):
        cluster_size = np.sum(labels == label)
        clusters_size[labels == label] = cluster_size

    if len(beta_hat.shape) == 1:

        beta_hat_degrouped = ward.inverse_transform(beta_hat) / clusters_size

    elif len(beta_hat.shape) == 2:

        n_features = pval_degrouped.shape[0]
        n_times = beta_hat.shape[1]
        beta_hat_degrouped = np.zeros((n_features, n_times))

        for i in range(n_times):

            beta_hat_degrouped[:, i] = (
                ward.inverse_transform(beta_hat[:, i]) / clusters_size
            )

    return (
        beta_hat_degrouped,
        pval_degrouped,
        pval_corr_degrouped,
        one_minus_pval_degrouped,
        one_minus_pval_corr_degrouped,
    )


def clustered_inference(
    X_init,
    y,
    ward,
    n_clusters,
    train_size=1.0,
    groups=None,
    method="desparsified-lasso",
    seed=0,
    n_jobs=1,
    memory=None,
    verbose=1,
    **kwargs,
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

    memory : str or joblib.Memory object, optional (default=None)
        Used to cache the output of the computation of the clustering
        and the inference. By default, no caching is done. If a string is
        given, it is the path to the caching directory.

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
    memory_instance = check_memory(memory=memory)

    n_samples, n_features = X_init.shape

    if verbose > 0:

        print(
            f"Clustered inference: n_clusters = {n_clusters}, "
            + f"inference method = {method}, seed = {seed}"
        )

    # Sampling
    train_index = _subsampling(n_samples, train_size, groups=groups, seed=seed)

    # Clustering
    X, ward = memory_instance.cache(_ward_clustering)(X_init, ward, train_index)

    # Preprocessing
    X = StandardScaler().fit_transform(X)
    y = y - np.mean(y)

    # Inference: computing reduced parameter vector and stats
    print("Clustered inference", kwargs)
    beta_hat_, pval_, pval_corr_, one_minus_pval_, one_minus_pval_corr_ = (
        memory_instance.cache(hd_inference, ignore=["n_jobs", "verbose", "memory"])(
            X, y, method, n_jobs=n_jobs, memory=memory, **kwargs
        )
    )

    # De-grouping
    beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = _degrouping(
        ward,
        beta_hat_,
        pval_,
        pval_corr_,
        one_minus_pval_,
        one_minus_pval_corr_,
    )

    return beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr
