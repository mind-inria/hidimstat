import numpy as np
from joblib import Parallel, delayed
from sklearn.base import check_is_fitted, clone
from sklearn.metrics import root_mean_squared_error
from typing import override

from hidimstat._utils.utils import _check_vim_predict_method
from hidimstat.base_variable_importance import (
    BaseVariableImportance,
    VariableImportanceFeatureGroup,
)


class LeaveOneCovariateIn(BaseVariableImportance, VariableImportanceFeatureGroup):
    def __init__(
        self,
        estimator,
        loss: callable = root_mean_squared_error,
        method: str = "predict",
        n_jobs: int = 1,
    ):
        """
        Leave One Covariate In.
        For more details, see the section 7.2 of :footcite:t:`ewald2024guide`.

        Parameters
        ----------
        estimator : sklearn compatible estimator, optional
            The estimator to use for the prediction.
        loss : callable, default=root_mean_squared_error
            The function to compute the loss when comparing the perturbed model
            to the original model.
        method : str, default="predict"
            The method used for making predictions. This determines the predictions
            passed to the loss function. Supported methods are "predict",
            "predict_proba", "decision_function", "transform".
        n_jobs : int, default=1
            The number of parallel jobs to run. Parallelization is done over the
            features or groups of features.
        """
        super().__init__()
        check_is_fitted(estimator)
        self.estimator = estimator
        self.loss = loss
        _check_vim_predict_method(method)
        self.method = method
        self.n_jobs = n_jobs
        # generated attributes
        self._list_univariate_model = []
        self.loss_reference_ = None

    @override
    def fit(self, X, y, features_groups=None):
        """
        Fit the marginal information variable importance model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        features_groups : dict, optional
            A dictionary where the keys are group identifiers and the values are lists
            of feature indices or names for each group. If None, each feature is
            treated as its own group.
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        super().fit(X, y, features_groups)
        X_ = np.asarray(X)
        y_ = np.asarray(y)

        # Parallelize the computation of the importance scores for each group
        self._list_univariate_model = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_fit_one_features_group)(X_, y_, features_groups_ids)
            for features_groups_ids in self._features_groups_ids
        )

    def predict(self, X):
        """
        Compute the predictions after perturbation of the data for each group of
        features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
            
        Returns
        -------
        out : array-like of shape (n_features_groups, n_samples)
            The predictions for each group of features.
        """
        self._check_fit(X)
        X_ = np.asarray(X)

        # Parallelize the computation of the importance scores for each group
        out_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_predict_one_features_group)(X_, features_group_id, features_groups_ids)
            for features_group_id, features_groups_ids in enumerate(self._features_groups_ids)
        )
        return np.array(out_list)

    def importance(self, X, y):
        """
        Compute the marginal importance scores for each group of features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
            
        Returns
        -------
        out_dict : dict
            A dictionary containing:
            - 'loss_reference' : float
            Loss of the original model predictions
            - 'loss' : dict
            Losses for each group's univariate predictions
            - 'importance' : ndarray of shape (n_features_groups,)
            Marginal importance scores for each feature group
        """
        self._check_fit(X)

        y_pred = self.predict(X)

        # reference to a dummy model
        if len(y_pred[0].shape) == 1 or y_pred[0].shape[1] == 1:
            # Regression: take the average value as reference
            y_ref = np.mean(y) * np.ones_like(y_pred[0])
            self.loss_reference_ = self.loss(y, y_ref)
        else:
            # Classification: take the most frequent value
            values, counts = np.unique(y, return_counts=True)
            y_ref = np.zeros_like(y_pred[0])
            y_ref[:, np.argmax(counts)] = 1.0
            self.loss_reference_ = self.loss(y, y_ref)

        self.importances_ = []
        for y_pred_j in y_pred:
            self.importances_.append(self.loss_reference_ - self.loss(y, y_pred_j))
        self.pvalues_ = None  # estimated pvlaue for method
        return self.importances_

    def fit_importance(self, X, y, cv, features_groups=None):
        """
        Fits the model to the data and computes feature importance.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The target values.
        cv :
            Cross-validation parameter.
        features_groups : dict, optional
            A dictionary where the keys are group identifiers and the values are lists
            of feature indices or names for each group. If None, each feature is
            treated as its own group.
            
        Returns
        -------
        importance : array-like
            The computed feature importance scores.
        """
        list_attribute_saved = ["importances_", "pvalues_", "_list_univariate_model"]
        save_value_attributes = []
        for train_index, test_index in cv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.fit(X_train, y_train, features_groups=features_groups)
            self.importance(X_test, y_test)
            save_value_attributes.append(
                [getattr(self, attribute) for attribute in list_attribute_saved]
            )
        # create an array of attributes:
        for attribute in list_attribute_saved:
            setattr(self, attribute, [])
        for value_attribute in save_value_attributes:
            for attribute, value in zip(list_attribute_saved, value_attribute):
                getattr(self, attribute).append(value)

        return np.mean(self.importances_, axis=0)

    def _joblib_fit_one_features_group(self, X, y, features_group_ids):
        """
        Helper function to fit a univariate model for a single group.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        features_group_ids : array-like
            The indices of features belonging to this group.
            
        Returns
        -------
        object
            The fitted univariate model for this group.
        """
        univariate_model = clone(self.estimator)
        return univariate_model.fit(X[:, features_group_ids].reshape(-1, len(features_group_ids)), y)

    def _joblib_predict_one_features_group(self, X, index_features_group, features_group_ids):
        """
        Helper function to predict for a single group.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        index_features_group : int
            The index of the group in _list_univariate_model.
        features_group_ids : array-like
            The indices of features belonging to this group.
            
        Returns
        -------
        float
            The prediction score for this group.
        """
        y_pred_loci = getattr(self._list_univariate_model[index_features_group], self.method)(
            X[:, features_group_ids].reshape(-1, len(features_group_ids))
        )
        return y_pred_loci