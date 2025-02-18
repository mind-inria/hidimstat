"""Base class for model agnostic variable importance measures based on perturbation."""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.metrics import root_mean_squared_error

from hidimstat.utils import _check_vim_predict_method


class BasePerturbation(BaseEstimator):
    def __init__(
        self,
        estimator,
        loss: callable = root_mean_squared_error,
        n_permutations: int = 50,
        method: str = "predict",
        n_jobs: int = 1,
    ):
        """
        Base class for model-agnostic variable importance measures based on
        perturbation.

        Parameters
        ----------
        estimator : object
            The model used for making predictions.
        loss : callable, default=root_mean_squared_error
            The function to compute the loss when comparing the perturbed model to the
            original model.
        n_permutations : int, default=50
            This parameter is relevant only for PermutationImportance or CPI.
            Specifies the number of times the variable group (residual for CPI) is
            permuted. For each permutation, the perturbed model's loss is calculated and
            averaged over all permutations.
        method : str, default="predict"
            The method used for making predictions. This determines the predictions
            passed to the loss function.
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

    def fit(self, X, y, groups=None):
        if groups is None:
            self.n_groups = X.shape[1]
            self.groups = {j: [j] for j in range(self.n_groups)}
            self._groups_ids = np.array(list(self.groups.values()), dtype=int)
        else:
            self.n_groups = len(groups)
            self.groups = groups
            if isinstance(X, pd.DataFrame):
                self._groups_ids = []
                for group_key in self.groups.keys():
                    self._groups_ids.append(
                        [
                            i
                            for i, col in enumerate(X.columns)
                            if col in self.groups[group_key]
                        ]
                    )
            else:
                self._groups_ids = np.array(list(self.groups.values()), dtype=int)

    def predict(self, X):
        self._check_fit()
        X_ = np.asarray(X)

        # Parallelize the computation of the importance scores for each group
        out_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_predict_one_group)(X_, group_id, group_key)
            for group_id, group_key in enumerate(self.groups.keys())
        )
        return np.stack(out_list, axis=0)

    def score(self, X, y):
        self._check_fit()

        out_dict = dict()

        y_pred = getattr(self.estimator, self.method)(X)
        loss_reference = self.loss(y, y_pred)
        out_dict["loss_reference"] = loss_reference

        y_pred = self.predict(X)
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

    def _check_fit(self):
        pass

    def _joblib_predict_one_group(self, X, group_id, group_key):
        group_ids = self._groups_ids[group_id]
        non_group_ids = np.delete(np.arange(X.shape[1]), group_ids)
        # Create an array X_perm_j of shape (n_permutations, n_samples, n_features)
        # where the j-th group of covariates is permuted
        X_perm = np.empty((self.n_permutations, X.shape[0], X.shape[1]))
        X_perm[:, :, non_group_ids] = np.delete(X, group_ids, axis=1)
        X_perm[:, :, group_ids] = self._permutation(X, group_id=group_id)
        # Reshape X_perm to allow for batch prediction
        X_perm_batch = X_perm.reshape(-1, X.shape[1])
        y_pred_perm = getattr(self.estimator, self.method)(X_perm_batch)

        # In case of classification, the output is a 2D array. Reshape accordingly
        if y_pred_perm.ndim == 1:
            y_pred_perm = y_pred_perm.reshape(self.n_permutations, X.shape[0])
        else:
            y_pred_perm = y_pred_perm.reshape(
                self.n_permutations, X.shape[0], y_pred_perm.shape[1]
            )
        return y_pred_perm

    def _permutation(self, X, group_id):
        raise NotImplementedError
