import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.metrics import root_mean_squared_error
import warnings

from hidimstat._utils.utils import _check_vim_predict_method


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
        assert n_permutations > 0, "n_permutations must be positive"
        assert n_jobs > 0, "n_jobs must be positive"
        self.estimator = estimator
        self.loss = loss
        _check_vim_predict_method(method)
        self.method = method
        self.n_jobs = n_jobs
        self.n_permutations = n_permutations
        self.n_groups = None

    def fit(self, X, y=None, groups=None):
        """Base fit method for perturbation-based methods. Identifies the groups.

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
        if groups is None:
            self.n_groups = X.shape[1]
            self.groups = {j: [j] for j in range(self.n_groups)}
            self._groups_ids = np.array(list(self.groups.values()), dtype=int)
        elif isinstance(groups, dict):
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
                self._groups_ids = [
                    np.array(ids, dtype=int) for ids in list(self.groups.values())
                ]
        else:
            raise ValueError("groups needs to be a dictionnary")

    def predict(self, X):
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
        self._check_fit(X)
        X_ = np.asarray(X)

        # Parallelize the computation of the importance scores for each group
        out_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_predict_one_group)(X_, group_id, group_key)
            for group_id, group_key in enumerate(self.groups.keys())
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
        self._check_fit(X)

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

    def _check_fit(self, X):
        """
        Check if the perturbation method has been properly fitted.

        This method verifies that the perturbation method has been fitted by checking
        if required attributes are set and if the number of features matches
        the grouped variables.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to validate against the fitted model.

        Raises
        ------
        ValueError
            If the method has not been fitted (i.e., if n_groups, groups,
            or _groups_ids attributes are missing).
        AssertionError
            If the number of features in X does not match the total number
            of features in the grouped variables.
        """
        if (
            self.n_groups is None
            or not hasattr(self, "groups")
            or not hasattr(self, "_groups_ids")
        ):
            raise ValueError(
                "The method is not fitted. The fit method must be called"
                " to set variable groups. If no grouping is needed,"
                " call fit with groups=None"
            )
        count = 0
        for group_id in self.groups.values():
            count += len(group_id)
        if X.shape[1] != count:
            warnings.warn("Not all features will has a importance score.")

    def _joblib_predict_one_group(self, X, group_id, group_key):
        """
        Compute the predictions after perturbation of the data for a given
        group of variables. This function is parallelized.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        group_id: int
            The index of the group of variables.
        group_key: str, int
            The key of the group of variables. (parameter use for debugging)
        """
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
        """Method for creating the permuted data for the j-th group of covariates."""
        raise NotImplementedError
