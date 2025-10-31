import numpy as np
from joblib import Parallel, delayed
from sklearn.base import check_is_fitted, clone
from sklearn.cluster import FeatureAgglomeration
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from hidimstat._utils.utils import check_random_state
from hidimstat.base_variable_importance import BaseVariableImportance
from hidimstat.desparsified_lasso import DesparsifiedLasso


class EnsembleClusteredInference(BaseVariableImportance):
    """
    Ensemble clustered inference algorithm for high-dimensional
    statistical inference.

    This algorithm implements the method described in :cite:`chevalier2022spatially` for

    This algorithm combines multiple runs of clustered inference with
    different random subsamples to provide more robust statistical estimates.
    It uses the desparsified lasso method for inference.

    Parameters
    ----------
    X_init : ndarray, shape (n_samples, n_features)
        Original high-dimensional input data matrix.

    y : ndarray, shape (n_samples,) or (n_samples, n_tasks)
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

    beta_hat : ndarray, shape (n_clusters,) or (n_clusters, n_tasks)
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

    def __init__(
        self,
        variable_importance=DesparsifiedLasso(),
        ward=None,
        n_bootstraps=25,
        scaler_sampling=None,
        train_size=1.0,
        groups=None,
        random_state=None,
        n_jobs=1,
        verbose=1,
    ):
        assert issubclass(DesparsifiedLasso, variable_importance.__class__)
        self.variable_importance = variable_importance
        assert ward is None or issubclass(
            FeatureAgglomeration, ward.__class__
        ), "Ward should a FeatureAgglomeration"
        self.ward = ward
        assert scaler_sampling is None or issubclass(
            StandardScaler, scaler_sampling.__class__
        )
        self.scaler_sampling = scaler_sampling
        self.n_bootstraps = n_bootstraps
        self.train_size = train_size
        self.groups = groups
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        # generalize to all the feature generated
        self.list_ward_scaler_vi_ = None
        self.list_importances_ = None
        self.list_pvalues_ = None
        self.list_pvalues_corr_ = None
        self.pvalues_corr_ = None

    def fit(self, X, y):
        rng = check_random_state(self.random_state)

        if self.verbose > 0:
            print(
                f"Clustered inference: n_clusters = {self.ward.n_clusters}, "
                + f"inference method desparsified lasso, seed = {self.random_state},"
                + f"groups = {self.groups is not None} "
            )
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        self.list_ward_scaler_vi_ = parallel(
            delayed(_bootstrap_run_fit)(
                X,
                y,
                self.train_size,
                self.groups,
                rng_spawn,
                self.ward,
                self.scaler_sampling,
                self.variable_importance,
            )
            for rng_spawn in rng.spawn(self.n_bootstraps)
        )
        return self

    def _check_fit(self):
        """
        Check if the model has been fit before performing analysis.

        This private method verifies that all necessary attributes have been set
        during the fitting process.
        These attributes include:
        - clf_x_
        - clf_y_
        - coefficient_
        - non_selection_

        Raises
        ------
        ValueError
            If any of the required attributes are missing, indicating the model
            hasn't been fit.
        """
        if self.list_ward_scaler_vi_ is None:
            raise ValueError("The  requires to be fit before any analysis")
        for ward, scaler, vi in self.list_ward_scaler_vi_:
            if ward is not None:
                try:
                    check_is_fitted(ward)
                except NotFittedError:
                    raise ValueError(
                        "The EnsembleClusteredInference requires to be fit before any analysis"
                    )
            if scaler is not None:
                try:
                    check_is_fitted(scaler)
                except NotFittedError:
                    raise ValueError(
                        "The EnsembleClusteredInference requires to be fit before any analysis"
                    )
            vi._check_fit()

    def importance(self, X, y):
        """
        Compute feature importance scores using distilled CRT.

        Calculates test statistics and p-values for each feature using residual
        correlations after the distillation process.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data matrix.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        importances_ : ndarray of shape (n_features,)
            Test statistics/importance scores for each feature. For unselected features,
            the score is set to 0.

        Attributes
        ----------
        importances_ : same as return value
        pvalues_ : ndarray of shape (n_features,)
            Two-sided p-values for each feature under Gaussian null.

        Notes
        -----
        For each selected feature j:
        1. Computes residuals from regressing X_j on other features
        2. Computes residuals from regressing y on other features
        3. Calculates test statistic from correlation of residuals
        4. Computes p-value assuming standard normal distribution
        """
        self._check_fit()

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        results = parallel(
            delayed(_bootstrap_run_importance)(
                ward,
                scaler,
                vi,
                X,
                y,
            )
            for ward, scaler, vi in self.list_ward_scaler_vi_
        )

        self.list_importances_ = []
        self.list_pvalues_ = []
        self.list_pvalues_corr_ = []
        for importances, pvalues, pvalues_corr in results:
            self.list_importances_.append(importances)
            self.list_pvalues_.append(pvalues)
            self.list_pvalues_corr_.append(pvalues_corr)

        self.importances_ = np.mean(self.list_importances_, axis=0)
        self.pvalues_ = np.mean(self.list_pvalues_, axis=0)
        self.pvalues_corr_ = np.mean(self.list_pvalues_corr_, axis=0)
        return self.importances_

    def fit_importance(self, X, y):
        """
        Fits the model to the data and computes feature importance.

        A convenience method that combines fit() and importance() into a single call.
        First fits the dCRT model to the data, then calculates importance scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data matrix.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        importance : ndarray of shape (n_features,)
            Feature importance scores/test statistics.
            For features not selected during screening, scores are set to 0.

        Notes
        -----
        Also sets the importances\_ and pvalues\_ attributes on the instance.
        See fit() and importance() for details on the underlying computations.
        """
        self.fit(X, y)
        return self.importance(X, y)


def _bootstrap_run_fit(
    X_init,
    y,
    train_size,
    groups,
    rng,
    ward,
    scaler_sampling,
    variable_importance,
):

    n_samples, n_features = X_init.shape

    ## This are the 3 step in first loop of the algorithm 2 of `chevalier2022spatially`
    # sampling row of X
    train_index = _subsampling(n_samples, train_size, groups=groups, random_state=rng)

    # transformation matrix
    if ward is not None:
        ward_ = clone(ward).fit(X_init[train_index, :])
        X_reduced = ward_.transform(X_init)
    else:
        X_reduced = X_init
        ward_ = None

    # Preprocessing
    if scaler_sampling is not None:
        scaler_sampling_ = clone(scaler_sampling)
        X_reduced = scaler_sampling_.fit_transform(X_reduced)
    else:
        scaler_sampling_ = None

    # inference methods
    variable_importance_ = clone(variable_importance).fit(X_reduced, y)

    return ward_, scaler_sampling_, variable_importance_


def _bootstrap_run_importance(ward_, scaler_sampling_, variable_importance_, X, y):
    # apply reduction
    if ward_ is not None:
        X_ = ward_.transform(X)
    else:
        X_ = X

    # apply Preprocessing
    if scaler_sampling_ is not None:
        X_ = scaler_sampling_.transform(X_)
    else:
        X_ = X

    variable_importance_.importance(X_, y)

    if ward_ is not None:
        pvalue = ward_.inverse_transform(variable_importance_.pvalues_)
        pvalue_corr = ward_.inverse_transform(variable_importance_.pvalues_corr_)
        importance = _ungroup_beta(
            variable_importance_.importances_, n_features=pvalue.shape[0], ward=ward_
        )
    else:
        pvalue = variable_importance_.pvalue_
        pvalue_corr = variable_importance_.pvalue_corr_
        importance = variable_importance_.importances_

    return importance, pvalue, pvalue_corr


def _subsampling(n_samples, train_size, groups=None, random_state=None):
    """
    Random subsampling for statistical inference.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the dataset.
    train_size : float
        Fraction of samples to include in the training set (between 0 and 1).
    groups : ndarray, shape (n_samples,), optional (default=None)
        Group labels for samples.
        If not None, a subset of groups is selected.
    random_state : int or None (default=None)
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
        random_state=np.random.RandomState(random_state.bit_generator),
    )
    if groups is not None:
        train_index = np.arange(n_samples)[np.isin(groups, train_index)]
    return train_index


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
        for i in range(n_tasks):
            beta_hat_degrouped[:, i] = (
                ward.inverse_transform(beta_hat[:, i]) / clusters_size
            )
    return beta_hat_degrouped
