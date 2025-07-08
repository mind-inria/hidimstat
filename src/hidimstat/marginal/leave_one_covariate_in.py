import numpy as np
import pandas as pd
import warnings
from joblib import Parallel, delayed
from sklearn.base import check_is_fitted, clone
from sklearn.metrics import root_mean_squared_error

from hidimstat._utils.utils import _check_vim_predict_method
from hidimstat.base_variable_importance import BaseVariableImportanceGroup


class LeaveOneCovariateIn(BaseVariableImportanceGroup):
    def __init__(
        self,
        estimator,
        loss: callable = root_mean_squared_error,
        method: str = "predict",
        n_jobs: int = 1,
    ):
        """Leave One Covariate In.

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
            variables or groups of variables.
        """
        super().__init__()
        check_is_fitted(estimator)
        self.estimator = estimator
        self.loss = loss
        _check_vim_predict_method(method)
        self.method = method
        self.n_jobs = n_jobs
        self.n_groups = None
        self._list_univariate_model = []
        self.loss_reference_ = None

    def fit(self, X, y, groups=None):
        """Fit the marginal information variable importance model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        groups : dict, optional
            A dictionary where the keys are group identifiers and the values are lists
            of feature indices or names for each group. If None, each feature is
            treated as its own group.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        super().fit(X, y, groups)
        X_ = np.asarray(X)
        y_ = np.asarray(y)

        # Parallelize the computation of the importance scores for each group
        self._list_univariate_model = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_fit_one_group)(X_, y_, groups_ids)
            for groups_ids in self._groups_ids
        )

    def predict(self, X):
        """
        Compute the predictions after perturbation of the data for each group of
        variables.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        out : array-like of shape (n_groups, n_samples)
            The predictions for each group of variables.
        """
        self._check_fit(X)
        X_ = np.asarray(X)

        # Parallelize the computation of the importance scores for each group
        out_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_predict_one_group)(X_, group_id, groups_ids)
            for group_id, groups_ids in enumerate(self._groups_ids)
        )
        return np.array(out_list)

    def importance(self, X, y):
        """
        Compute the marginal importance scores for each group of variables.

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
            - 'importance' : ndarray of shape (n_groups,)
            Marginal importance scores for each variable group
        """
        self._check_fit(X)

        y_pred = getattr(self.estimator, self.method)(X)
        self.loss_reference_ = self.loss(y, y_pred)

        y_pred = self.predict(X)
        self.importances_ = []
        for y_pred_j in y_pred:
            self.importances_.append(self.loss(y, y_pred_j))
        self.pvalues_ = None  # estimated pvlaue for method
        return self.importances_

    def fit_importance(self, X, y, cv=None):
        """
        Fits the model to the data and computes feature importance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The target values.
        cv : None or int, optional (default=None)
            (not used) Cross-validation parameter.
            A warning will be issued if provided.

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
            self.fit(X_train, y_train)
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

    def _joblib_fit_one_group(self, X, y, group_ids):
        """Helper function to fit a univariate model for a single group.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        group_ids : array-like
            The indices of features belonging to this group.

        Returns
        -------
        object
            The fitted univariate model for this group.
        """
        univariate_model = clone(self.estimator)
        return univariate_model.fit(X[:, group_ids].reshape(-1, 1), y)

    def _joblib_predict_one_group(self, X, index_group, group_ids):
        """Helper function to predict for a single group.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        index_group : int
            The index of the group in _list_univariate_model.
        group_ids : array-like
            The indices of features belonging to this group.

        Returns
        -------
        float
            The prediction score for this group.
        """
        y_pred_loci = getattr(self._list_univariate_model[index_group], self.method)(
            X[:, group_ids].reshape(-1, 1)
        )
        return y_pred_loci
