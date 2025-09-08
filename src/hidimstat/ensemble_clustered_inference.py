import warnings

import numpy as np
from sklearn.base import clone
from joblib import Parallel, delayed
from sklearn.utils.validation import check_memory
from sklearn.exceptions import NotFittedError
from sklearn.cluster import FeatureAgglomeration
from sklearn.base import check_is_fitted
from sklearn.utils import check_random_state

from hidimstat.desparsified_lasso import DesparsifiedLasso
from hidimstat.base_variable_importance import BaseVariableImportance
from hidimstat._utils.bootstrap import _subsampling


def _ungroup_beta(beta_hat, n_features, ward):
    """
    Ungroup cluster-level beta coefficients to individual feature-level
    coefficients.

    Parameters
    ----------
    beta_hat : ndarray, shape (n_clusters,) or (n_clusters, n_times)
        Beta coefficients at cluster level
    n_features : int
        Number of features in original space
    ward : sklearn.cluster.FeatureAgglomeration
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
    ward : sklearn.cluster.FeatureAgglomeration
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


class ClusteredInference(BaseVariableImportance):
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

    beta_hat : ndarray, shape (n_clusters,) or (n_clusters, n_times)
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
        ward,
        n_clusters,
        variable_importance=DesparsifiedLasso(),
        scaler_sampling=None,
        train_size=1.0,
        groups=None,
        seed=0,
        n_jobs=1,
        memory=None,
        verbose=1,
    ):
        self.ward = ward
        self.n_clusters = n_clusters
        self.variable_importance = variable_importance
        self.scaler_sampling = scaler_sampling
        self.train_size = train_size
        self.groups = groups
        self.seed = seed
        self.n_jobs = n_jobs
        self.memory = memory
        self.verbose = verbose

        # generalize to all the feature generated
        self.pvalues_corr_ = None

    def fit(self, X_init, y):
        memory = check_memory(memory=self.memory)
        assert issubclass(
            self.ward.__class__, FeatureAgglomeration
        ), "ward need to an instance of sklearn.cluster.FeatureAgglomeration"

        n_samples, n_features = X_init.shape

        if self.verbose > 0:
            print(
                f"Clustered inference: n_clusters = {self.n_clusters}, "
                + f"inference method desparsified lasso, seed = {self.seed},"
                + f"groups = {self.groups is not None} "
            )

        ## This are the 3 step in first loop of the algorithm 2 of [1]
        # sampling row of X
        train_index = _subsampling(
            n_samples, self.train_size, groups=self.groups, seed=self.seed
        )

        # transformation matrix
        X_reduced, self.ward = memory.cache(_ward_clustering)(
            X_init, clone(self.ward), train_index
        )

        # Preprocessing
        if self.scaler_sampling is not None:
            self.scaler_sampling = clone(self.scaler_sampling)
            X_reduced = self.scaler_sampling.fit_transform(X_reduced)

        # inference methods
        self.variable_importance = memory.cache(self.variable_importance.fit)(
            X_reduced,
            y,
        )
        return self

<<<<<<< HEAD
    # inference methods
    if hasattr(kwargs, "lasso_cv") and kwargs["lasso_cv"] is not None:
        pass
    elif len(y.shape) > 1 and y.shape[1] > 1:
        kwargs["lasso_cv"] = MultiTaskLassoCV(
            eps=1e-2,
            fit_intercept=False,
            cv=KFold(n_splits=5, shuffle=True, random_state=0),
            tol=1e-4,
            max_iter=5000,
            random_state=1,
            n_jobs=1,
        )
    else:
        kwargs["lasso_cv"] = LassoCV(
            eps=1e-2,
            fit_intercept=False,
            cv=KFold(n_splits=5, shuffle=True, random_state=0),
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
||||||| parent of ec9ff4e (Add Encldel and Cluster)
    # inference methods
    multitasklassoCV = MultiTaskLassoCV(
        eps=1e-2,
        fit_intercept=False,
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        tol=1e-4,
        max_iter=5000,
        random_state=1,
        n_jobs=1,
    )
    lasso_cv = LassoCV(
        eps=1e-2,
        fit_intercept=False,
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        tol=1e-4,
        max_iter=5000,
        random_state=1,
        n_jobs=1,
    )
    desparsified_lassos = memory.cache(
        DesparsifiedLasso(
            lasso_cv=(
                multitasklassoCV if len(y.shape) > 1 and y.shape[1] > 1 else lasso_cv
            ),
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
=======
    def _check_fit(self):
        """
        Check if the model has been fit before performing analysis.
>>>>>>> ec9ff4e (Add Encldel and Cluster)

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
        self.variable_importance._check_fit()
        try:
            check_is_fitted(self.ward)
            if self.scaler_sampling is not None:
                check_is_fitted(self.scaler_sampling)
        except NotFittedError:
            raise ValueError(
                "The ClusteredInference requires to be fit before any analysis"
            )

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
        X_reduced = self.ward.transform(X)
        if self.scaler_sampling is not None:
            X_reduced = self.scaler_sampling.transform(X_reduced)

        self.variable_importance.importance(X_reduced, y)
        beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = _degrouping(
            self.ward,
            self.variable_importance.importances_,
            self.variable_importance.pvalues_,
            self.variable_importance.pvalues_corr_,
            1 - self.variable_importance.pvalues_,
            1 - self.variable_importance.pvalues_corr_,
        )

        self.importances_ = beta_hat
        self.pvalues_ = pval
        self.pvalues_corr_ = pval_corr
        return self.importances_

    def fit_importance(self, X, y, cv=None):
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
        cv : None or int, optional (default=None)
            Not used. Included for compatibility. A warning will be issued if provided.

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
        if cv is not None:
            warnings.warn("cv won't be used")
        self.fit(X, y)
        return self.importance(X, y)


class EnsembleClusteredInference(BaseVariableImportance):
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

    def __init__(
        self,
        variable_importance,
        n_bootstraps=25,
        n_jobs=None,
        verbose=1,
        memory=None,
        random_state=None,
    ):
        self.variable_importance = variable_importance
        self.n_bootstraps = n_bootstraps
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.memory = memory
        self.random_state = random_state

        self.list_variable_importances_ = None

    def fit(self, X, y):
        """
        Fit the dCRT model.

        This method fits the Distilled Conditional Randomization Test (DCRT) model
        as described in :footcite:t:`liu2022fast`. It performs optional feature
        screening using Lasso, computes coefficients, and prepares the model for
        importance and p-value computation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data matrix.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the fitted instance.

        Notes
        -----
        Main steps:
        1. Optional data centering with StandardScaler
        2. Lasso screening of variables (if no estimated coefficients provided)
        3. Feature selection based on coefficient magnitudes
        4. Model refitting on selected features (if refit=True)
        5. Fit model for future distillation

        The screening threshold controls which features are kept based on their
        Lasso coefficients. Features with coefficients below the threshold are
        set to zero.

        References
        ----------
        .. footbibliography::
        """
        rng = check_random_state(self.random_state)
        seed = rng.randint(1)

        def run_fit(variable_importance, X, y, random_state):
            return variable_importance(random_state=random_state, n_jobs=1).fit(X, y)

        self.list_variable_importances_ = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose
        )(
            delayed(run_fit)(clone(self.variable_importance), X, y, i)
            for i in np.arange(seed, seed + self.n_bootstraps)
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
        if self.list_variable_importances_ is None:
            raise ValueError("The D0CRT requires to be fit before any analysis")

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

        def run_importance(variable_importance, X, y):
            variable_importance.importance(X, y)
            return None

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        parallel(
            delayed(run_importance)(variable_importance, X, y)
            for variable_importance in self.list_variable_importances_
        )

        # Ensembling
        # TODO check if selection_FDR is good
        self.importances_ = np.mean(
            [vi.importances_ for vi in self.list_variable_importances_], axis=0
        )
        # pvalue selection
        self.pvalues_ = np.array(
            [vi.pvalues_ for vi in self.list_variable_importances_]
        )
        return self.importances_

    def fit_importance(self, X, y, cv=None):
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
        cv : None or int, optional (default=None)
            Not used. Included for compatibility. A warning will be issued if provided.

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
        if cv is not None:
            warnings.warn("cv won't be used")
        self.fit(X, y)
        return self.importance(X, y)
