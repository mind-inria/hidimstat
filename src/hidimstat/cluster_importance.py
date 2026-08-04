import inspect

import numpy as np
from sklearn.base import clone
from sklearn.cluster import FeatureAgglomeration
from sklearn.utils.validation import check_memory

from hidimstat._utils.utils import check_random_state
from hidimstat.base_variable_importance import BaseVariableImportance


class ClusterImportance(BaseVariableImportance):
    """
    Clustered inference with any variable importance method.

    This algorithm computes a single clustered inference on groups of features
    using an arbitrary variable importance measure for statistical inference.

    Parameters
    ----------
    vim: hidimstat.BaseVariableImportance
        An instance of any variable importance method that derives from hidimstat's BaseVariableImportance.
    clustering: sklearn.cluster.FeatureAgglomeration
        An instance of a clustering method that operates on features.

    Attributes
    ----------
    vim_ : hidimstat.BaseVariableImportance
        Fitted variable importance method that will be clustered.
    clustering_ : sklearn.cluster.FeatureAgglomeration
        Fitted clustering object.
    clustering_samples_ : ndarray, (n_samples*cluster_frac,)
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
        vim,
        clustering,
    ):
        self.vim = vim
        self.clustering = clustering

        self.vim_ = None
        self.clustering_ = None

    def fit(self, X, y):
        """
        Fit the clustering and variable importance method on the data.

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
        assert issubclass(self.vim.__class__, BaseVariableImportance), (
            "estimator needs to be a subclass of BaseVariableImportance"
        )
        assert issubclass(self.clustering.__class__, FeatureAgglomeration), (
            "clustering needs to be an instance of sklearn.cluster.FeatureAgglomeration"
        )

        self.n_features_in_ = X.shape[1]
        self.clustering_ = self.clustering.fit(X)
        X_reduced = self.clustering_.transform(X)

        self.vim_ = clone(self.vim)
        self.vim_.estimator.fit(X_reduced, y)
        self.vim_ = self.vim_.fit(X_reduced, y)
        return self

    def importance(self, X, y):
        """
        Compute feature importance from the underlying variable importance method.
        Then map the importance scores from cluster level back to feature level.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data matrix.
        y : ndarray, shape (n_samples,) or (n_samples, n_tasks)
            Target variable(s).
        """
        self._check_fit()

        X_reduced = self.clustering_.transform(X)
        self.vim_.importance(X_reduced, y)

        self.pvalues_ = self.clustering_.inverse_transform(self.vim_.pvalues_)

        self.importances_ = self._ungroup_importance(
            self.vim_.importances_,
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

    @staticmethod
    def _ungroup_importance(importance, n_features, ward):
        """
        Ungroup cluster-level importances to individual feature-level
        importances.

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
        # degroup importances
        if len(importance.shape) == 1:
            # weighting the weight of importance with the size of the cluster
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

    def _check_fit(self):
        """
        Check that an estimator has been fitted after removing each group of
        covariates.
        """
        if self.vim_ is None:
            raise ValueError(
                "The estimator needs to be fit before using them."
            )
