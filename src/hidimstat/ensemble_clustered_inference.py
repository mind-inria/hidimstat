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
    Random subsampling for statistical inference.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the dataset.
    train_size : float
        Fraction of samples to include in the training set (between 0 and 1).
    groups : ndarray, shape (n_samples,), optional (default=None)
        Group labels for samples. If provided, subsampling is done
        at group level.
    seed : int, optional (default=0)
        Random seed for reproducibility.

    Returns
    -------
    train_index : ndarray
        Indices of selected samples for training.
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


def _rescale_beta(beta_hat, n_features, ward):
    """
    Rescales cluster-level beta coefficients to individual feature-level
    coefficients.

    Parameters
    ----------
    beta_hat : ndarray, shape (n_clusters,) or (n_clusters, n_times)
        Beta coefficients at cluster level
    n_features : int
        Number of features in original space
    ward : AgglomerativeClustering
        Fitted clustering object

    Returns
    -------
    beta_hat_degrouped : ndarray, shape (n_features,) or (n_features, n_times)
        Rescaled beta coefficients for individual features, weighted by
        inverse cluster size

    Notes
    -----
    Each coefficient is scaled by 1/cluster_size to maintain proper magnitude
    when distributing cluster effects to individual features.
    Handles both univariate (1D) and multivariate (2D) beta coefficients.
    """
    labels = ward.labels_
    # compute the size of each cluster
    clusters_size = np.zeros(labels.size)
    for label in range(labels.max() + 1):
        cluster_size = np.sum(labels == label)
        clusters_size[labels == label] = cluster_size
    # degroup beta_hat
    if len(beta_hat.shape) == 1:
        # weighting the weight of beta with the size of the cluster
        beta_hat_degrouped = ward.inverse_transform(beta_hat) / clusters_size
    elif len(beta_hat.shape) == 2:
        n_times = beta_hat.shape[1]
        beta_hat_degrouped = np.zeros((n_features, n_times))
        for i in range(n_times):
            beta_hat_degrouped[:, i] = (
                ward.inverse_transform(beta_hat[:, i]) / clusters_size
            )
    return beta_hat_degrouped


def _degrouping(ward, beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr):
    """
    Degroup and rescale cluster-level statistics to individual features.
    This function takes cluster-level statistics and assigns them back
    to individual features, while appropriately rescaling the parameter
    estimates based on cluster sizes.

    Parameters
    ----------
    ward : AgglomerativeClustering
        Fitted clustering object containing the hierarchical structure
    beta_hat : ndarray, shape (n_clusters,) or (n_clusters, n_times)
        Estimated parameters at cluster level
    pval : ndarray, shape (n_clusters,)
        P-values at cluster level
    pval_corr : ndarray, shape (n_clusters,)
        Corrected p-values at cluster level
    one_minus_pval : ndarray, shape (n_clusters,)
        1 - p-values at cluster level
    one_minus_pval_corr : ndarray, shape (n_clusters,)
        1 - corrected p-values at cluster level
    Returns
    -------
    beta_hat : ndarray, shape (n_features,) or (n_features, n_times)
        Rescaled parameter estimates for individual features
    pval : ndarray, shape (n_features,)
        P-values for individual features
    pval_corr : ndarray, shape (n_features,)
        Corrected p-values for individual features
    one_minus_pval : ndarray, shape (n_features,)
        1 - p-values for individual features
    one_minus_pval_corr : ndarray, shape (n_features,)
        1 - corrected p-values for individual features
    Notes
    -----
    The beta_hat values are rescaled by dividing by the cluster size
    to maintain the proper scale of the estimates when moving from
    cluster-level to feature-level.
    The function handles both 1D and 2D beta_hat arrays for single and
    multiple time points.
    """
    # degroup variable other than beta_hat
    pval, pval_corr, one_minus_pval, one_minus_pval_corr = map(
        ward.inverse_transform,
        [pval, pval_corr, one_minus_pval, one_minus_pval_corr],
    )

    beta_hat = _rescale_beta(beta_hat, n_features=pval.shape[0], ward=ward)

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
    **kwargs,
):
    """
    Clustered inference algorithm for statistical analysis of
    high-dimensional data.

    This algorithm implements the method described in :cite:`chevalier2022spatially` for
    performing statistical inference on high-dimensional linear models
    using feature clustering to reduce dimensionality.

    Parameters
    ----------
    X_init : ndarray, shape (n_samples, n_features)
        Original high-dimensional input data matrix.

    y : ndarray, shape (n_samples,) or (n_samples, n_times)
        Target variable(s). Can be univariate or multivariate (temporal) data.

    ward : sklearn.cluster.FeatureAgglomeration
        Hierarchical clustering object that implements Ward's method for
        feature agglomeration.

    n_clusters : int
        Number of clusters to use for dimensionality reduction.

    scaler_sampling : sklearn.preprocessing object, optional (default=None)
        Scaler to standardize the clustered features.

    train_size : float, optional (default=1.0)
        Fraction of samples to use for computing the clustering.
        When train_size=1.0, all samples are used.

    groups : ndarray, shape (n_samples,), optional (default=None)
        Sample group labels for stratified subsampling.

    seed : int, optional (default=0)
        Random seed for reproducible subsampling.

    n_jobs : int, optional (default=1)
        Number of parallel jobs for computation.

    verbose : int, optional (default=1)
        Verbosity level for progress messages.

    **kwargs : dict
        Additional arguments passed to the statistical inference function.

    Returns
    -------
    ward_ : FeatureAgglomeration
        Fitted clustering object.

    beta_hat : ndarray, shape (n_clusters,) or (n_clusters, n_times)
        Estimated coefficients at cluster level.

    theta_hat : ndarray
        Estimated precision matrix.

    omega_diag : ndarray
        Diagonal of the covariance matrix.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    The algorithm follows these main steps:
    1. Subsample the data (if train_size < 1)
    2. Cluster features using Ward hierarchical clustering
    3. Transform data to cluster space
    4. Perform statistical inference using desparsified lasso
    """
    n_samples, n_features = X_init.shape

    if verbose > 0:
        print(
            f"Clustered inference: n_clusters = {n_clusters}, "
            + f"inference method desparsified lasso, seed = {seed},"
            + f"groups = {groups is not None} "
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
        group=len(y.shape) > 1 and y.shape[1] > 1,  # detection of multiOutput
        n_jobs=n_jobs,
        verbose=verbose,
        **kwargs,
    )

    return ward_, beta_hat, theta_hat, omega_diag


def clustered_inference_pvalue(
    n_samples, group, ward, beta_hat, theta_hat, omega_diag, **kwargs
):
    """
    Compute corrected p-values at the cluster level and transform them
    back to feature level.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset
    group : bool
        If True, uses group lasso p-values for multivariate outcomes
    ward : AgglomerativeClustering
        Fitted clustering object
    beta_hat : ndarray
        Estimated coefficients at cluster level
    theta_hat : ndarray
        Estimated precision matrix
    omega_diag : ndarray
        Diagonal elements of the covariance matrix
    **kwargs : dict
        Additional arguments passed to p-value computation functions

    Returns
    -------
    beta_hat : ndarray
        Degrouped coefficients at feature level
    pval : ndarray
        P-values for each feature
    pval_corr : ndarray
        Multiple testing corrected p-values
    one_minus_pval : ndarray
        1 - p-values for numerical stability
    one_minus_pval_corr : ndarray
        1 - corrected p-values
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
    **kwargs,
):
    """
    Ensemble clustered inference algorithm for high-dimensional
    statistical inference, as described in :cite:`chevalier2022spatially`.

    This algorithm combines multiple runs of clustered inference with
    different random subsamples to provide more robust statistical estimates.
    It uses the desparsified lasso method for inference.

    Parameters
    ----------
    X_init : ndarray, shape (n_samples, n_features)
        Original high-dimensional input data matrix.

    y : ndarray, shape (n_samples,) or (n_samples, n_times)
        Target variable(s). Can be univariate or multivariate (temporal) data.

    ward : sklearn.cluster.FeatureAgglomeration
        Feature agglomeration object implementing Ward hierarchical clustering.

    n_clusters : int
        Number of clusters for dimensionality reduction.

    scaler_sampling : sklearn.preprocessing object, optional (default=None)
        Scaler to standardize the clustered features.

    train_size : float, optional (default=0.3)
        Fraction of samples used for clustering. Using train_size < 1 enables
        random subsampling for better generalization.

    groups : ndarray, shape (n_samples,), optional (default=None)
        Sample group labels for stratified subsampling. Ensures balanced
        representation of groups in subsamples.

    seed : int, optional (default=0)
        Random seed for reproducible subsampling sequence.

    n_bootstraps : int, optional (default=25)
        Number of bootstrap iterations for ensemble inference.

    n_jobs : int or None, optional (default=None)
        Number of parallel jobs. None means using all processors.

    verbose : int, optional (default=1)
        Verbosity level for progress reporting.

    **kwargs : dict
        Additional keyword arguments passed to statistical inference functions.

    Returns
    -------
    list_ward : list of FeatureAgglomeration
        List of fitted clustering objects from each bootstrap.

    list_beta_hat : list of ndarray
        List of estimated coefficients from each bootstrap.

    list_theta_hat : list of ndarray
        List of estimated precision matrices.

    list_omega_diag : list of ndarray
        List of diagonal elements of covariance matrices.

    Notes
    -----
    The algorithm performs these steps for each bootstrap iteration:
    1. Subsample the data using stratified sampling if groups are provided
    2. Cluster features using Ward's hierarchical clustering
    3. Transform data to reduced cluster space
    4. Perform statistical inference using desparsified lasso
    5. Aggregate results across all iterations

    References
    ----------
    .. footbibliography::
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
            **kwargs,
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
    Compute and aggregate p-values across multiple bootstrap iterations
    using an aggregation method.

    This function performs statistical inference on each bootstrap sample
    and combines the results using a specified aggregation method to obtain
    robust estimates.
    The implementation follows the methodology in :footcite:`chevalier2022spatially`.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset
    group : bool
        If True, uses group lasso p-values for multivariate outcomes
    list_ward : list of AgglomerativeClustering
        List of fitted clustering objects from bootstraps
    list_beta_hat : list of ndarray
        List of estimated coefficients at cluster level from each bootstrap
    list_theta_hat : list of ndarray
        List of estimated precision matrices from each bootstrap
    list_omega_diag : list of ndarray
        List of diagonal elements of covariance matrices from each bootstrap
    aggregate_method : callable, default=aggregate_quantiles
        Function to aggregate results across bootstraps. Must accept a 2D array
        and return a 1D array of aggregated values.
    n_jobs : int or None, optional (default=None)
        Number of parallel jobs. None means using all processors.
    verbose : int, default=0
        Verbosity level for computation progress
    **kwargs : dict
        Additional arguments passed to p-value computation functions

    Returns
    -------
    beta_hat : ndarray, shape (n_features,) or (n_features, n_times)
        Averaged coefficients across bootstraps
    pval : ndarray, shape (n_features,)
        Aggregated p-values for each feature
    pval_corr : ndarray, shape (n_features,)
        Aggregated multiple testing corrected p-values
    one_minus_pval : ndarray, shape (n_features,)
        Aggregated 1-p values for numerical stability
    one_minus_pval_corr : ndarray, shape (n_features,)
        Aggregated 1-corrected p values for numerical stability

    References
    ----------
    [1] footbibliography::
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
