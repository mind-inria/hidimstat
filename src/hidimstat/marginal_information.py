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
        """

        Parameters
        ----------
        estimator : sklearn compatible estimator, optional
            The estimator to use for the prediction.
        loss : callable, default=root_mean_squared_error
            The function to compute the loss when comparing the perturbed model
            to the original model.
        n_permutations : int, default=50
            This parameter is relevant only for PFI or CPI.
            Specifies the number of times the variable group (residual for CPI) is
            permuted. For each permutation, the perturbed model's loss is calculated
            and averaged over all permutations.
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
        """

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        y: array-like of shape (n_samples,)
            Not used, only present for consistency with the sklearn API.
        groups: dict, optional
            A dictionary where the keys are the group names and the values are the
            list of column names corresponding to each group. If None, the groups are
            identified based on the columns of X.
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
        X: array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        out: array-like of shape (n_groups, n_permutations, n_samples)
            The predictions after perturbation of the data for each group of variables.
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
        Compute the importance scores for each group of covariates.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        y: array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        out_dict: dict
            A dictionary containing the following keys:
            - 'loss_reference': the loss of the model with the original data.
            - 'loss': a dictionary containing the loss of the perturbed model
            for each group.
            - 'importance': the importance scores for each group.
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
        """Check that the estimator has been fitted if needed."""
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

    def _joblib_predict_one_group(self, X, y, index_group, group_ids):
        """

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        group_id: int
            The index of the group of variables.
        group_key: str, int
            The key of the group of variables. (parameter use for debugging)
        """
        univariate_model = self._list_univariate_model[index_group]
        return univariate_model.score(X[:, group_ids].reshape(-1, 1), y)

    def _joblib_fit_one_group(self, X, y, group_ids):
        """

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        group_id: int
            The index of the group of variables.
        group_key: str, int
            The key of the group of variables. (parameter use for debugging)
        """
        univariate_model = clone(self.estimator)
        univariate_model.fit(X[:, group_ids].reshape(-1, 1), y)
        return univariate_model
