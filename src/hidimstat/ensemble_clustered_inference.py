import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.cluster import FeatureAgglomeration
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_memory
from tqdm import tqdm

from hidimstat._utils.bootstrap import _subsampling
from hidimstat._utils.utils import check_random_state
from hidimstat.desparsified_lasso import DesparsifiedLasso
from hidimstat.statistical_tools.aggregation import quantile_aggregation
from hidimstat.statistical_tools.multiple_testing import fdr_threshold


def _ungroup_beta(beta_hat, n_features, ward):
    """
    Ungroup cluster-level beta coefficients to individual feature-level
    coefficients.

    Parameters
    ----------
    beta_hat : ndarray, shape (n_clusters,) or (n_clusters, n_tasks)
        Beta coefficients at cluster level
    n_features : int
        Number of features in original space
    ward : sklearn.cluster.FeatureAgglomeration
        Fitted clustering object

    Returns
    -------
    beta_hat_degrouped : ndarray, shape (n_features,) or (n_features, n_tasks)
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
        n_tasks = beta_hat.shape[1]
        beta_hat_degrouped = np.zeros((n_features, n_tasks))
        for i in range(n_task):
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
    ward : sklearn.cluster.FeatureAgglomeration
        Fitted clustering object containing the hierarchical structure
    beta_hat : ndarray, shape (n_clusters,) or (n_clusters, n_task)
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
    beta_hat : ndarray, shape (n_features,) or (n_features, n_task)
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

    beta_hat = _ungroup_beta(beta_hat, n_features=pval.shape[0], ward=ward)

    return beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr


def _ward_clustering(X_init, ward, train_index):
    """
    Performs Ward clustering on data using a training subset.

    This function applies Ward hierarchical clustering to a dataset, where the clustering
    is computed based on a subset of samples but then applied to the full dataset.

    Parameters
    ----------
    X_init : numpy.ndarray
        Initial data matrix of shape (n_samples, n_features) to be clustered
    ward : sklearn.cluster.FeatureAgglomeration
        Ward clustering estimator instance
    train_index : array-like
        Indices of samples to use for computing the clustering

    Returns
    -------
    tuple
        - X_reduced : numpy.ndarray
            Transformed data matrix after applying Ward clustering
        - ward : sklearn.cluster.FeatureAgglomeration
            Fitted Ward clustering estimator
    """
    ward = ward.fit(X_init[train_index, :])
    X_reduced = ward.transform(X_init)
    return X_reduced, ward


def clustered_inference(
    X_init,
    y,
    ward,
    scaler_sampling=None,
    train_size=1.0,
    groups=None,
    random_state=None,
    n_jobs=1,
    memory=None,
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

    y : ndarray, shape (n_samples,) or (n_samples, n_task)
        Target variable(s). Can be univariate or multivariate (temporal) data.

    ward : sklearn.cluster.FeatureAgglomeration
        Hierarchical clustering object that implements Ward's method for
        feature agglomeration.

    scaler_sampling : sklearn.preprocessing object, optional (default=None)
        Scaler to standardize the clustered features.

    train_size : float, optional (default=1.0)
        Fraction of samples to use for computing the clustering.
        When train_size=1.0, all samples are used.

    groups : ndarray, shape (n_samples,), optional (default=None)
        Sample group labels for stratified subsampling.

    random_state : int, optional (default=None)
        Random seed for reproducible subsampling.

    n_jobs : int, optional (default=1)
        Number of parallel jobs for computation.

    memory : str or joblib.Memory object, optional (default=None)
        Used to cache the output of the computation of the clustering
        and the inference. By default, no caching is done. If a string is
        given, it is the path to the caching directory.

    verbose : int, optional (default=1)
        Verbosity level for progress messages.

    **kwargs : dict
        Additional arguments passed to the statistical inference function.

    Returns
    -------
    ward_ : FeatureAgglomeration
        Fitted clustering object.

    beta_hat : ndarray, shape (n_clusters,) or (n_clusters, n_task)
        Estimated coefficients at cluster level.

    theta_hat : ndarray
        Estimated precision matrix.

    precision_diag : ndarray
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
    memory = check_memory(memory=memory)
    rng = check_random_state(random_state)
    assert issubclass(
        ward.__class__, FeatureAgglomeration
    ), "ward need to an instance of sklearn.cluster.FeatureAgglomeration"

    n_samples, n_features = X_init.shape

    ## This are the 3 step in first loop of the algorithm 2 of [1]
    # sampling row of X
    train_index = _subsampling(n_samples, train_size, groups=groups, random_state=rng)

    # transformation matrix
    X_reduced, ward_ = memory.cache(_ward_clustering)(X_init, clone(ward), train_index)

    # Preprocessing
    if scaler_sampling is not None:
        X_reduced = clone(scaler_sampling).fit_transform(X_reduced)

    # inference methods
    if hasattr(kwargs, "lasso_cv") and kwargs["lasso_cv"] is not None:
        pass
    elif len(y.shape) > 1 and y.shape[1] > 1:
        kwargs["model_y"] = MultiTaskLassoCV(
            eps=1e-2,
            fit_intercept=False,
            cv=KFold(n_splits=5),
            tol=1e-4,
            max_iter=5000,
            random_state=1,
            n_jobs=1,
        )
    else:
        kwargs["model_y"] = LassoCV(
            eps=1e-2,
            fit_intercept=False,
            cv=KFold(n_splits=5),
            tol=1e-4,
            max_iter=5000,
            random_state=1,
            n_jobs=1,
        )

    desparsified_lassos = memory.cache(
        DesparsifiedLasso(
            n_jobs=n_jobs,
            memory=memory,
            verbose=verbose,
            **kwargs,
        ).fit,
        ignore=["n_jobs", "verbose", "memory"],
    )(
        X_reduced,
        y,
    )
    desparsified_lassos.importance(X_reduced, y)

    return ward_, desparsified_lassos


def clustered_inference_pvalue(n_samples, group, ward, desparsified_lassos, **kwargs):
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
    precision_diag : ndarray
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

    # De-grouping
    beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = _degrouping(
        ward,
        desparsified_lassos.importances_,
        desparsified_lassos.pvalues_,
        desparsified_lassos.pvalues_corr_,
        1 - desparsified_lassos.pvalues_,
        1 - desparsified_lassos.pvalues_corr_,
    )

    return beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr


def ensemble_clustered_inference(
    X_init,
    y,
    ward,
    scaler_sampling=None,
    train_size=0.3,
    groups=None,
    random_state=None,
    n_bootstraps=25,
    n_jobs=None,
    verbose=1,
    memory=None,
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

    y : ndarray, shape (n_samples,) or (n_samples, n_task)
        Target variable(s). Can be univariate or multivariate (temporal) data.

    ward : sklearn.cluster.FeatureAgglomeration
        Feature agglomeration object implementing Ward hierarchical clustering.

    scaler_sampling : sklearn.preprocessing object, optional (default=None)
        Scaler to standardize the clustered features.

    train_size : float, optional (default=0.3)
        Fraction of samples used for clustering. Using train_size < 1 enables
        random subsampling for better generalization.

    groups : ndarray, shape (n_samples,), optional (default=None)
        Sample group labels for stratified subsampling. Ensures balanced
        representation of groups in subsamples.

    inference_method : str, optional (default='desparsified-lasso')
        Method used for inference.
        Currently, the two available methods are 'desparsified-lasso'
        and 'group-desparsified-lasso'. Use 'desparsified-lasso' for
        non-temporal data and 'group-desparsified-lasso' for temporal data.

    random_seed: int, optional (default=None)
        Seed used for generating the first random subsample of the data.
        This seed controls the clustering randomness.

    ensembling_method : str, optional (default='quantiles')
        Method used for ensembling. Currently, the two available methods
        are 'quantiles' and 'median'.

    gamma_min : float, optional (default=0.2)
        Lowest gamma-quantile considered to compute the adaptive
        quantile aggregation formula. This parameter is used only if
        `ensembling_method` is 'quantiles'.

    n_bootstraps : int, optional (default=25)
        Number of bootstrap iterations for ensemble inference.

    n_jobs : int or None, optional (default=None)
        Number of parallel jobs. None means using all processors.

    verbose: int, optional (default=1)
        The verbosity level. If `verbose > 0`, a message is printed before
        running the clustered inference.

    memory : joblib.Memory or str, optional (default=None)
        Used to cache the output of the clustering and inference computation.
        By default, no caching is done. If provided, it should be the path
        to the caching directory or a joblib.Memory object.

    **kwargs : dict
        Additional keyword arguments passed to statistical inference functions.

    Returns
    -------
    list_ward : list of FeatureAgglomeration
        List of fitted clustering objects from each bootstrap.

    list_beta_hat : list of ndarray
        List of estimated coefficients from each bootstrap.

    pval : ndarray, shape (n_features,)
        p-value, with numerically accurate values for
        positive effects (i.e., for p-values close to zero).

    list_theta_hat : list of ndarray
        List of estimated precision matrices.

    list_precision_diag : list of ndarray
        List of diagonal elements of covariance matrices.

    one_minus_pval : ndarray, shape (n_features,)
        One minus the p-value, with numerically accurate values
        for negative effects (i.e., for p-values close to one).

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
    memory = check_memory(memory=memory)
    assert issubclass(
        ward.__class__, FeatureAgglomeration
    ), "ward need to an instance of sklearn.cluster.FeatureAgglomeration"
    rng = check_random_state(random_state)

    # Clustered inference algorithms
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(clustered_inference)(
            X_init,
            y,
            clone(ward),
            scaler_sampling=scaler_sampling,
            train_size=train_size,
            groups=groups,
            random_state=spawned_state,
            n_jobs=1,
            verbose=verbose,
            memory=memory,
            **kwargs,
        )
        for spawned_state in tqdm(rng.spawn(n_bootstraps), disable=(verbose == 0))
    )
    list_ward, list_desparsified_lassos = [], []
    for ward, desparsified_lassos in results:
        list_ward.append(ward)
        list_desparsified_lassos.append(desparsified_lassos)
    return list_ward, list_desparsified_lassos


def ensemble_clustered_inference_pvalue(
    n_samples,
    group,
    list_ward,
    list_desparsified_lassos,
    fdr=0.1,
    fdr_control="bhq",
    reshaping_function=None,
    adaptive_aggregation=False,
    gamma=0.5,
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
    list_precision_diag : list of ndarray
        List of diagonal elements of covariance matrices from each bootstrap
    fdr : float, default=0.1
        False discovery rate threshold for multiple testing correction
    fdr_control : str, default="bhq"
        Method for FDR control ('bhq' for Benjamini-Hochberg)
        Available methods are:
        * 'bhq': Standard Benjamini-Hochberg :footcite:`benjamini1995controlling,bhy_2001`
        * 'bhy': Benjamini-Hochberg-Yekutieli :footcite:p:`bhy_2001`
        * 'ebh': e-Benjamini-Hochberg :footcite:`wang2022false`
    reshaping_function : callable, optional (default=None)
        Function to reshape data before FDR control
    adaptive_aggregation : bool, default=False
        Whether to use adaptive quantile aggregation
    gamma : float, default=0.5
        Quantile level for aggregation
    n_jobs : int or None, optional (default=None)
        Number of parallel jobs. None means using all processors.
    verbose : int, default=0
        Verbosity level for computation progress
    **kwargs : dict
        Additional arguments passed to p-value computation functions

    Returns
    -------
    beta_hat : ndarray, shape (n_features,) or (n_features, n_task)
        Averaged coefficients across bootstraps
    selected : ndarray, shape (n_features,)
        Selected features: 1 for positive effects, -1 for negative effects,
        0 for non-selected features

    References
    ----------
    .. footbibliography::
    """
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(clustered_inference_pvalue)(
            n_samples,
            group,
            list_ward[i],
            list_desparsified_lassos[i],
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
    # pvalue selection
    aggregated_pval = quantile_aggregation(
        np.array(list_pval), gamma=gamma, adaptive=adaptive_aggregation
    )
    threshold_pval = fdr_threshold(
        aggregated_pval,
        fdr=fdr,
        method=fdr_control,
        reshaping_function=reshaping_function,
    )
    # 1-pvalue selection
    aggregated_one_minus_pval = quantile_aggregation(
        np.array(list_one_minus_pval), gamma=gamma, adaptive=adaptive_aggregation
    )
    threshold_one_minus_pval = fdr_threshold(
        aggregated_one_minus_pval,
        fdr=fdr,
        method=fdr_control,
        reshaping_function=reshaping_function,
    )
    # group seelction
    selected = np.zeros_like(beta_hat)
    selected[np.where(aggregated_pval <= threshold_pval)] = 1
    selected[np.where(aggregated_one_minus_pval <= threshold_one_minus_pval)] = -1

    return beta_hat, selected
