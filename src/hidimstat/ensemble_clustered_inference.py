import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.cluster import FeatureAgglomeration
from sklearn.utils import resample
from sklearn.utils.validation import check_memory
from tqdm import tqdm

from hidimstat._utils.utils import check_random_state
from hidimstat.base_variable_importance import BaseVariableImportance
from hidimstat.desparsified_lasso import DesparsifiedLasso
from hidimstat.statistical_tools.aggregation import quantile_aggregation


class CluDL(BaseVariableImportance):
    """
    Clustered inference with desparsified lasso.

    This algorithm computes a single clustered inference on groups of features
    using the desparsified lasso method for statistical inference.

    Parameters
    ----------
    clustering: sklearn.cluster.FeatureAgglomeration
        An instance of a clustering method that operates on features.
    cluster_boostrap_size: float, optional (default=1.0)
        Fraction of samples used for computing the clustering.
        When cluster_boostrap_size=1.0, all samples are used.
    bootstrap_groups: ndarray, shape (n_samples,), optional (default=None)
        Sample group labels for stratified subsampling.
    random_state: int, optional (default=None)
        Random seed for reproducible subsampling.
    memory : joblib.Memory or str, optional (default=None)
        Used to cache the output of the clustering and inference computation.
        By default, no caching is done. If provided, it should be the path
        to the caching directory or a joblib.Memory object.
    **kwargs :
        Additional parameters passed to the DesparsifiedLasso constructor.

    Attributes
    ----------
    desparsified_lasso_ : DesparsifiedLasso
        Fitted desparsified lasso estimator.
    clustering_ : sklearn.cluster.FeatureAgglomeration
        Fitted clustering object.
    clustering_samples_ : ndarray, (n_samples*cluster_boostrap_size,)
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
        cluster_boostrap_size=1.0,
        bootstrap_groups=None,
        random_state=None,
        memory=None,
        **kwargs,
    ):
        assert issubclass(
            clustering.__class__, FeatureAgglomeration
        ), "clustering need to be an instance of sklearn.cluster.FeatureAgglomeration"
        self.desparsified_lasso = DesparsifiedLasso(random_state=random_state, **kwargs)
        self.clustering = clustering
        self.cluster_boostrap_size = cluster_boostrap_size
        self.bootstrap_groups = bootstrap_groups
        self.random_state = random_state
        self.memory = memory

        self.desparsified_lasso_ = None
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
        self : CluDL
            Fitted estimator.
        """
        memory = check_memory(memory=self.memory)
        rng = check_random_state(self.random_state)

        self.n_features_in_ = X.shape[1]

        # Clustering
        self.clustering_samples_ = self._subsampling(
            n_samples=X.shape[0],
            train_size=self.cluster_boostrap_size,
            groups=self.bootstrap_groups,
            random_state=rng,
        )
        self.clustering_ = self.clustering.fit(X[self.clustering_samples_, :])
        X_reduced = self.clustering_.transform(X)

        # Desparsified lasso inference
        self.desparsified_lasso.random_state = self.random_state
        self.desparsified_lasso_ = memory.cache(self.desparsified_lasso.fit)(
            X_reduced, y
        )

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

        self.desparsified_lasso_.importance()

        self.pvalues_ = self.clustering_.inverse_transform(
            self.desparsified_lasso_.pvalues_
        )

        self.importances_ = self._ungroup_beta(
            self.desparsified_lasso_.importances_,
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
        self : CluDL
            Fitted estimator with computed importances.
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


class EnCluDL(BaseVariableImportance):
    """
    Ensemble clustered inference with desparsified lasso. Performs multiple runs
    of clustered inference using different clustering obtained from random subsamples
    of the data. The results are then aggregated to provide robust feature importance
    scores and p-values. This algorithm is based on the method described in
    :footcite:`chevalier2022spatially`.

    Parameters
    ----------
    desparsified_lasso: DesparsifiedLasso
        An instance of the DesparsifiedLasso class for statistical inference.
    clustering: sklearn.cluster.FeatureAgglomeration
        An instance of a clustering method that operates on features.
    n_bootstraps: int, optional (default=25)
        Number of bootstrap iterations for ensemble inference.
    cluster_boostrap_size: float, optional (default=0.3)
        Fraction of samples used for computing the clustering.
        When cluster_boostrap_size=1.0, all samples are used.
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
    clustering_desparsified_lassos_ : list of DesparsifiedLasso
        List of fitted CluDL estimators from each bootstrap.
    importances_ : ndarray, shape (n_features,) or (n_features, n_tasks)
        Estimated coefficients at feature level.
    pvalues_ : ndarray, shape (n_features,)
        P-values for each feature.

    .. footbibliography::
    """

    def __init__(
        self,
        desparsified_lasso,
        clustering,
        n_bootstraps=25,
        cluster_boostrap_size=0.3,
        bootstrap_groups=None,
        n_jobs=1,
        random_state=None,
        memory=None,
        ensembling_method="quantiles",
        gamma=0.5,
        adaptive_aggregation=False,
    ):
        self.desparsified_lasso = desparsified_lasso
        self.clustering = clustering
        self.n_bootstraps = n_bootstraps
        self.cluster_boostrap_size = cluster_boostrap_size
        self.bootstrap_groups = bootstrap_groups
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.memory = memory
        self.ensembling_method = ensembling_method
        self.gamma = gamma
        self.adaptive_aggregation = adaptive_aggregation

        self.desparsified_lassos_ = None

    @staticmethod
    def _joblib_fit_one(
        desparsified_lasso,
        clustering,
        cluster_boostrap_size,
        bootstrap_groups,
        X,
        y,
        random_state,
        memory,
    ):
        clu_dl = CluDL(
            desparsified_lasso=desparsified_lasso,
            clustering=clustering,
            cluster_boostrap_size=cluster_boostrap_size,
            bootstrap_groups=bootstrap_groups,
            random_state=random_state,
            memory=memory,
        )
        clu_dl.fit(X, y)
        return clu_dl

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
        self : EnCluDL
            Fitted estimator.
        """
        rng = check_random_state(self.random_state)

        self.clustering_desparsified_lassos_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_fit_one)(
                desparsified_lasso=self.desparsified_lasso,
                clustering=self.clustering,
                cluster_boostrap_size=self.cluster_boostrap_size,
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
        """
        for i in tqdm(
            range(self.n_bootstraps),
            desc="Computing importances",
            total=self.n_bootstraps,
        ):
            self.clustering_desparsified_lassos_[i].importance()

        self.importances_ = np.mean(
            [clu_dl.importances_ for clu_dl in self.clustering_desparsified_lassos_],
            axis=0,
        )

        self.pvalues_ = quantile_aggregation(
            np.array(
                [clu_dl.pvalues_ for clu_dl in self.clustering_desparsified_lassos_]
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
            Estimated coefficients at feature level.
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
