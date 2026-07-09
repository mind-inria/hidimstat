import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.cluster import FeatureAgglomeration
from sklearn.utils import resample
from sklearn.utils.validation import check_memory
from tqdm import tqdm

from hidimstat._utils.utils import check_random_state
from hidimstat.base_variable_importance import BaseVariableImportance
from hidimstat.statistical_tools.aggregation import quantile_aggregation


class CluVI(BaseVariableImportance):
    """
    Clustered inference with any variable importance method.

    This algorithm computes a single clustered inference on groups of features
    using the desparsified lasso method for statistical inference.

    Parameters
    ----------
    clustering: sklearn.cluster.FeatureAgglomeration
        An instance of a clustering method that operates on features.
    vi_estimator: hidimstat.BaseVariableImportance
        An instance of any variable importance estimator that derives from hidimstat's BaseVariableImportance.
    cluster_bootstrap_size: float, optional (default=1.0)
        Fraction of samples used for computing the clustering.
        When cluster_bootstrap_size=1.0, all samples are used.
    bootstrap_groups: ndarray, shape (n_samples,), optional (default=None)
        Sample group labels for stratified subsampling.
    random_state: int, optional (default=None)
        Random seed for reproducible subsampling.
    memory : joblib.Memory or str, optional (default=None)
        Used to cache the output of the clustering and inference computation.
        By default, no caching is done. If provided, it should be the path
        to the caching directory or a joblib.Memory object.

    Attributes
    ----------
    vi_estimator_ : hidimstat.BaseVariableImportance
        Fitted variable importance estimator.
    clustering_ : sklearn.cluster.FeatureAgglomeration
        Fitted clustering object.
    clustering_samples_ : ndarray, (n_samples*cluster_bootstrap_size,)
        Indices of samples used for clustering.
    importances_ : ndarray, shape (n_clusters,) or (n_clusters, n_tasks)
        Estimated coefficients at cluster level.
    pvalues_ : ndarray, shape (n_clusters,)
        P-values for each cluster.
    n_features_ : int
        Number of features in the original data.

    """

    def __init__(
        self,
        clustering,
        vi_estimator=BaseVariableImportance(),
        cluster_bootstrap_size=1.0,
        bootstrap_groups=None,
        random_state=None,
        memory=None,
    ):
        assert issubclass(vi_estimator.__class__, BaseVariableImportance), (
            "vi_estimator need to be a subclass of BaseVariableImportance"
        )
        assert issubclass(clustering.__class__, FeatureAgglomeration), (
            "clustering need to be an instance of sklearn.cluster.FeatureAgglomeration"
        )
        self.vi_estimator = clone(vi_estimator)
        self.clustering = clone(clustering)
        self.cluster_bootstrap_size = cluster_bootstrap_size
        self.bootstrap_groups = bootstrap_groups
        self.random_state = random_state
        self.memory = memory

        self.vi_estimator_ = None
        self.clustering_ = None
        self.clustering_samples_ = None

    def fit(self, X, y):
        """
        Fit the clustering and desparsified lasso on the data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data matrix.
        y : ndarray, shape (n_samples,) or (n_samples, n_tasks)
            Target variable(s).

        Returns
        -------
        self : CluVI
            Fitted estimator.
        """
        memory = check_memory(memory=self.memory)
        rng = check_random_state(self.random_state)

        self.n_features_in_ = X.shape[1]

        # Clustering
        self.clustering_samples_ = self._subsampling(
            n_samples=X.shape[0],
            train_size=self.cluster_bootstrap_size,
            groups=self.bootstrap_groups,
            random_state=rng,
        )
        self.clustering_ = self.clustering.fit(X[self.clustering_samples_, :])
        X_reduced = self.clustering_.transform(X)

        # Desparsified lasso inference
        if hasattr(self.vi_estimator, "random_state"):
            self.vi_estimator.random_state = self.random_state
        self.vi_estimator_ = self.vi_estimator.fit(X_reduced, y)
        return self

    def importance(self, X=None, y=None):
        """
        Compute feature importance using desparsified lasso. Then map the importance
        scores from cluster level back to feature level.

        Parameters
        ----------
        X :
            Not used, present for API consistency by convention.
        y :
            Not used, present for API consistency by convention.
        """
        del y
        del X
        self.vi_estimator_.importance()

        self.pvalues_ = self.clustering_.inverse_transform(
            self.vi_estimator_.pvalues_
        )

        self.importances_ = self._ungroup_importance(
            self.vi_estimator_.importances_,
            n_features=self.n_features_in_,
            ward=self.clustering_,
        )
        return self.importances_

    def fit_importance(self, X, y):
        """
        Fit the model and compute feature importance.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data matrix.
        y : ndarray, shape (n_samples,) or (n_samples, n_tasks)
            Target variable(s).

        Returns
        -------
        importances_ : ndarray, shape (n_features,) or (n_features, n_tasks)
            Estimated importance values at feature level.
        """
        self.fit(X, y)
        self.importance(X, y)
        return self.importances_

    def fdr_selection(
        self,
        fdr,
        fdr_control="bhq",
        reshaping_function=None,
        two_tailed_test=True,
    ):
        """
        Overrides the signature to set two_tailed_test=True by default.
        """
        return super().fdr_selection(
            fdr=fdr,
            fdr_control=fdr_control,
            reshaping_function=reshaping_function,
            two_tailed_test=two_tailed_test,
        )

    @staticmethod
    def _ungroup_importance(importance, n_features, ward):
        """
        Ungroup cluster-level beta coefficients to individual feature-level
        coefficients.

        Parameters
        ----------
        importance : ndarray, shape (n_clusters,) or (n_clusters, n_tasks)
            Importance values at cluster level
        n_features : int
            Number of features in original space
        ward : sklearn.cluster.FeatureAgglomeration
            Fitted clustering object

        Returns
        -------
        importance_degrouped : ndarray, shape (n_features,) or (n_features, n_tasks)
            Rescaled importance values for individual features, weighted by
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
            clusters_size[labels == label] = np.sum(labels == label)
        # degroup beta_hat
        if len(importance.shape) == 1:
            # weighting the weight of beta with the size of the cluster
            importance_degrouped = (
                ward.inverse_transform(importance) / clusters_size
            )
        elif len(importance.shape) == 2:
            n_tasks = importance.shape[1]
            importance_degrouped = np.zeros((n_features, n_tasks))
            for i in range(n_tasks):
                importance_degrouped[:, i] = (
                    ward.inverse_transform(importance[:, i]) / clusters_size
                )
        return importance_degrouped

    @staticmethod
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
        random_state : int, optional (default=0)
            Random seed for reproducibility.

        Returns
        -------
        train_index : ndarray
            Indices of selected samples for training.
        """
        index_row = (
            np.arange(n_samples) if groups is None else np.unique(groups)
        )
        train_index = resample(
            index_row,
            n_samples=int(len(index_row) * train_size),
            replace=False,
            random_state=np.random.RandomState(random_state.bit_generator),
        )
        if groups is not None:
            train_index = np.arange(n_samples)[np.isin(groups, train_index)]
        return train_index


class EnCluVI(BaseVariableImportance):
    """
    Ensemble clustered inference with desparsified lasso. Performs multiple runs
    of clustered inference using different clustering obtained from random subsamples
    of the data. The results are then aggregated to provide robust feature importance
    scores and p-values. This algorithm is based on the method described in
    :footcite:`chevalier2022spatially`.

    Parameters
    ----------
    vi_estimator: hidimstat.BaseVariableImportance
        Any variable importance method that derives from hidimstat's BaseVariableImportance class.
    clustering: sklearn.cluster.FeatureAgglomeration
        An instance of a clustering method that operates on features.
    n_bootstraps: int, optional (default=25)
        Number of bootstrap iterations for ensemble inference.
    cluster_bootstrap_size: float, optional (default=0.3)
        Fraction of samples used for computing the clustering.
        When cluster_bootstrap_size=1.0, all samples are used.
    bootstrap_groups: ndarray, shape (n_samples,), optional (default=None)
        Sample group labels for stratified subsampling.
    n_jobs : int or None, optional (default=1)
        Number of parallel jobs.
    random_state: int, optional (default=None)
        Random seed for reproducible subsampling.
    memory : joblib.Memory or str, optional (default=None)
        Used to cache the output of the clustering and inference computation.
        By default, no caching is done. If provided, it should be the path
        to the caching directory or a joblib.Memory object.
    ensembling_method : str, optional (default='quantiles')
        Method used for ensembling. Currently, the two available methods
        are 'quantiles' and 'median'.
    gamma : float, optional (default=0.2)
        Lowest gamma-quantile considered to compute the adaptive
        quantile aggregation formula. This parameter is used only if
        `ensembling_method` is 'quantiles'.
    adaptive_aggregation : bool, optional (default=True)
        Whether to use adaptive quantile aggregation when `ensembling_method`
        is 'quantiles'.

    Attributes
    ----------
    clustering_vi_estimators_ : list of hidimstat.BaseVariableImportance
        List of fitted CluVI estimators from each bootstrap.
    importances_ : ndarray, shape (n_features,) or (n_features, n_tasks)
        Estimated coefficients at feature level.
    pvalues_ : ndarray, shape (n_features,)
        P-values for each feature.

    .. footbibliography::
    """

    def __init__(
        self,
        vi_estimator,
        clustering,
        n_bootstraps=25,
        cluster_bootstrap_size=0.3,
        bootstrap_groups=None,
        n_jobs=1,
        random_state=None,
        memory=None,
        ensembling_method="quantiles",
        gamma=0.5,
        adaptive_aggregation=False,
    ):
        self.vi_estimator = vi_estimator
        self.clustering = clustering
        self.n_bootstraps = n_bootstraps
        self.cluster_bootstrap_size = cluster_bootstrap_size
        self.bootstrap_groups = bootstrap_groups
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.memory = memory
        self.ensembling_method = ensembling_method
        self.gamma = gamma
        self.adaptive_aggregation = adaptive_aggregation

        self.vi_estimator_ = None

    @staticmethod
    def _joblib_fit_one(
        vi_estimator,
        clustering,
        cluster_bootstrap_size,
        bootstrap_groups,
        X,
        y,
        random_state,
        memory,
    ):
        clu_vi = CluVI(
            vi_estimator=vi_estimator,
            clustering=clustering,
            cluster_bootstrap_size=cluster_bootstrap_size,
            bootstrap_groups=bootstrap_groups,
            random_state=random_state,
            memory=memory,
        )
        clu_vi.fit(X, y)
        return clu_vi

    def fit(self, X, y):
        """
        Fit multiple clustered inferences on random subsamples of the data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data matrix.
        y : ndarray, shape (n_samples,) or (n_samples, n_tasks)
            Target variable(s).

        Returns
        -------
        self : EnCluVI
            Fitted estimator.
        """
        rng = check_random_state(self.random_state)

        self.clustering_vi_estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_fit_one)(
                vi_estimator=clone(self.vi_estimator),
                clustering=clone(self.clustering),
                cluster_bootstrap_size=self.cluster_bootstrap_size,
                bootstrap_groups=self.bootstrap_groups,
                X=X,
                y=y,
                random_state=rng_spawned,
                memory=self.memory,
            )
            for rng_spawned in tqdm(
                rng.spawn(self.n_bootstraps),
                desc="Fitting clustered inferences",
                total=self.n_bootstraps,
            )
        )

        return self

    def importance(self, X=None, y=None):
        """
        Compute feature importance by aggregating results from multiple
        clustered inferences.

        Parameters
        ----------
        X :
            Not used, present for API consistency by convention.
        y :
            Not used, present for API consistency by convention.

        Returns
        -------
        importances_ : ndarray, shape (n_features,) or (n_features, n_tasks)
            Estimated importance values at feature level.
        """
        del y
        del X
        for i in tqdm(
            range(self.n_bootstraps),
            desc="Computing importances",
            total=self.n_bootstraps,
        ):
            self.clustering_vi_estimators_[i].importance()

        self.importances_ = np.mean(
            [clu_dl.importances_ for clu_dl in self.clustering_vi_estimators_],
            axis=0,
        )

        self.pvalues_ = quantile_aggregation(
            np.array(
                [clu_dl.pvalues_ for clu_dl in self.clustering_vi_estimators_]
            ),
            gamma=self.gamma,
            adaptive=self.adaptive_aggregation,
        )

        return self.importances_

    def fit_importance(self, X, y):
        """
        Fit the model and compute feature importance.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data matrix.
        y : ndarray, shape (n_samples,) or (n_samples, n_tasks)
            Target variable(s).

        Returns
        -------
        importances_ : ndarray, shape (n_features,) or (n_features, n_tasks)
            Estimated importance values at feature level.
        """
        self.fit(X, y)
        self.importance(X, y)
        return self.importances_

    def fdr_selection(
        self,
        fdr,
        fdr_control="bhq",
        reshaping_function=None,
        two_tailed_test=True,
    ):
        """
        Overrides the signature to set two_tailed_test=True by default.
        """
        return super().fdr_selection(
            fdr=fdr,
            fdr_control=fdr_control,
            reshaping_function=reshaping_function,
            two_tailed_test=two_tailed_test,
        )
