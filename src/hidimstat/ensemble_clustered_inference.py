import numpy as np
from joblib import Parallel, delayed
from sklearn.utils.validation import check_memory

from hidimstat.clustered_inference import clustered_inference
from hidimstat.statistical_tools.multi_sample_split import (
    aggregate_medians,
    aggregate_quantiles,
)


def _ensembling(
    list_beta_hat,
    list_pval,
    list_pval_corr,
    list_one_minus_pval,
    list_one_minus_pval_corr,
    method="quantiles",
    gamma_min=0.2,
):

    beta_hat = np.asarray(list_beta_hat)
    list_pval = np.asarray(list_pval)
    list_pval_corr = np.asarray(list_pval_corr)
    list_one_minus_pval = np.asarray(list_one_minus_pval)
    list_one_minus_pval_corr = np.asarray(list_one_minus_pval_corr)

    beta_hat = np.mean(list_beta_hat, axis=0)

    if method == "quantiles":

        pval = aggregate_quantiles(list_pval, gamma_min)
        pval_corr = aggregate_quantiles(list_pval_corr, gamma_min)
        one_minus_pval = aggregate_quantiles(list_one_minus_pval, gamma_min)
        one_minus_pval_corr = aggregate_quantiles(list_one_minus_pval_corr, gamma_min)

    elif method == "medians":

        pval = aggregate_medians(list_pval)
        pval_corr = aggregate_medians(list_pval_corr)
        one_minus_pval = aggregate_medians(list_one_minus_pval)
        one_minus_pval_corr = aggregate_medians(list_one_minus_pval_corr)

    else:

        raise ValueError("Unknown ensembling method.")

    return beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr


def ensemble_clustered_inference(
    X_init,
    y,
    ward,
    n_clusters,
    train_size=0.3,
    groups=None,
    inference_method="desparsified-lasso",
    seed=0,
    ensembling_method="quantiles",
    gamma_min=0.2,
    n_bootstraps=25,
    n_jobs=1,
    memory=None,
    verbose=1,
    **kwargs,
):
    """Ensemble clustered inference algorithm.

    For more details, see :footcite:t:`chevalier2021decoding`.

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
        If `train_size = 1`, clustering is not random since all samples
        are used to compute the clustering.

    groups : ndarray, shape (n_samples,), optional (default=None)
        Group labels for each sample. If not None, `groups` is used to build
        the subsamples that serve to compute the clustering.

    inference_method : str, optional (default='desparsified-lasso')
        Method used for inference.
        Currently, the two available methods are 'desparsified-lasso'
        and 'group-desparsified-lasso'. Use 'desparsified-lasso' for
        non-temporal data and 'group-desparsified-lasso' for temporal data.

    seed: int, optional (default=0)
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
        Number of clustered inference algorithm solutions to compute before
        ensembling.

    n_jobs : int or None, optional (default=1)
        Number of CPUs used to compute several clustered inference
        algorithms simultaneously.

    memory : joblib.Memory or str, optional (default=None)
        Used to cache the output of the clustering and inference computation.
        By default, no caching is done. If provided, it should be the path
        to the caching directory or a joblib.Memory object.

    verbose: int, optional (default=1)
        The verbosity level. If `verbose > 0`, a message is printed before
        running the clustered inference.

    **kwargs:
        Arguments passed to the statistical inference function.

    Returns
    -------
    beta_hat : ndarray, shape (n_features,) or (n_features, n_times)
        Estimated parameter vector or matrix.

    pval : ndarray, shape (n_features,)
        p-value, with numerically accurate values for
        positive effects (i.e., for p-values close to zero).

    pval_corr : ndarray, shape (n_features,)
        p-value corrected for multiple testing.

    one_minus_pval : ndarray, shape (n_features,)
        One minus the p-value, with numerically accurate values
        for negative effects (i.e., for p-values close to one).

    one_minus_pval_corr : ndarray, shape (n_features,)
        One minus the p-value corrected for multiple testing.

    References
    ----------
    .. footbibliography::
    """

    memory = check_memory(memory=memory)

    # Clustered inference algorithms
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(clustered_inference)(
            X_init,
            y,
            ward,
            n_clusters,
            train_size=train_size,
            groups=groups,
            method=inference_method,
            seed=i,
            n_jobs=1,
            memory=memory,
            verbose=verbose,
            **kwargs,
        )
        for i in np.arange(seed, seed + n_bootstraps)
    )

    # Collecting results
    list_beta_hat = []
    list_pval, list_pval_corr = [], []
    list_one_minus_pval, list_one_minus_pval_corr = [], []

    for i in range(n_bootstraps):

        list_beta_hat.append(results[i][0])
        list_pval.append(results[i][1])
        list_pval_corr.append(results[i][2])
        list_one_minus_pval.append(results[i][3])
        list_one_minus_pval_corr.append(results[i][4])

    # Ensembling
    beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = _ensembling(
        list_beta_hat,
        list_pval,
        list_pval_corr,
        list_one_minus_pval,
        list_one_minus_pval_corr,
        method=ensembling_method,
        gamma_min=gamma_min,
    )

    return beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr
