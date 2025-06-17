import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, check_is_fitted, clone
from sklearn.metrics import root_mean_squared_error

from hidimstat._utils.utils import _check_vim_predict_method


class MarginalImportance(BaseEstimator):
    def __init__(
        self,
        estimator,
        loss: callable = root_mean_squared_error,
        n_permutations: int = 50,
        method: str = "predict",
        n_jobs: int = 1,
    ):
        """Marginal Information Variable Importance.

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
        check_is_fitted(estimator)
        self.estimator = estimator
        self.loss = loss
        _check_vim_predict_method(method)
        self.method = method
        self.n_jobs = n_jobs
        self.n_permutations = n_permutations
        self.n_groups = None
        self._list_univariate_model = []

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
        X_ = np.asarray(X)
        y_ = np.asarray(y)

        ###########################################
        # same as base permutation
        if groups is None:
            self.n_groups = X_.shape[1]
            self.groups = {j: [j] for j in range(self.n_groups)}
            self._groups_ids = np.array(list(self.groups.values()), dtype=int)
        else:
            self.n_groups = len(groups)
            self.groups = groups
            if isinstance(X_, pd.DataFrame):
                self._groups_ids = []
                for group_key in self.groups.keys():
                    self._groups_ids.append(
                        [
                            i
                            for i, col in enumerate(X_.columns)
                            if col in self.groups[group_key]
                        ]
                    )
            else:
                self._groups_ids = [
                    np.array(ids, dtype=int) for ids in list(self.groups.values())
                ]
        ###########################################

        # Parallelize the computation of the importance scores for each group
        self._list_univariate_model = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_fit_one_group)(X_, y_, groups_ids)
            for groups_ids in self._groups_id.values()
        )

    def predict(self, X, y):
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
        self._check_fit()
        X_ = np.asarray(X)
        y_ = np.asarray(y)

        # Parallelize the computation of the importance scores for each group
        out_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_predict_one_group)(X_, y_, group_id, groups_ids)
            for group_id, groups_ids in enumerate(self._groups_ids.values())
        )
        return np.stack(out_list, axis=0)

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
        ###########################################
        # same as base permutation
        self._check_fit()

        out_dict = dict()

        y_pred = getattr(self.estimator, self.method)(X)
        loss_reference = self.loss(y, y_pred)
        out_dict["loss_reference"] = loss_reference

        y_pred = self.predict(X, y)
        out_dict["loss"] = dict()
        for j, y_pred_j in enumerate(y_pred):
            list_loss = []
            for y_pred_perm in y_pred_j:
                list_loss.append(self.loss(y, y_pred_perm))
            out_dict["loss"][j] = np.array(list_loss)

        out_dict["importance"] = np.array(
            [
                np.mean(out_dict["loss"][j]) - loss_reference
                for j in range(self.n_groups)
            ]
        )
        return out_dict
        ######################################

    def _check_fit(self):
        """Check that the estimator has been fitted if needed.

        Checks if the estimator instance has the required attributes to ensure it has been properly fitted.
        Specifically verifies:
        - n_groups is set
        - groups attribute exists
        - _groups_ids attribute exists
        - _list_univariate_model is not empty

        Raises
        ------
        ValueError
            If any of the required attributes are missing, indicating the estimator has not been fitted

        """
        if (
            self.n_groups is None
            or not hasattr(self, "groups")
            or not hasattr(self, "_groups_ids")
            or len(self._list_univariate_model) == 0
        ):
            raise ValueError(
                "The estimator is not fitted. The fit method must be called"
                " to set variable groups. If no grouping is needed,"
                " call fit with groups=None"
            )

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
        univariate_model.fit(X[:, group_ids].reshape(-1, 1), y)
        return univariate_model

    def _joblib_predict_one_group(self, X, y, index_group, group_ids):
        """Helper function to predict for a single group.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        index_group : int
            The index of the group in _list_univariate_model.
        group_ids : array-like
            The indices of features belonging to this group.

        Returns
        -------
        float
            The prediction score for this group.
        """
        univariate_model = self._list_univariate_model[index_group]
        return univariate_model.score(X[:, group_ids].reshape(-1, 1), y)
