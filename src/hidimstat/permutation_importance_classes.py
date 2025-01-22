import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, check_is_fitted, clone
from sklearn.metrics import root_mean_squared_error

from hidimstat.utils import _check_vim_predict_method


class BasePermutation(BaseEstimator):
    def __init__(
        self,
        estimator,
        loss: callable = root_mean_squared_error,
        n_permutations: int = 50,
        method: str = "predict",
        n_jobs: int = 1,
    ):
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
        else:
            self.n_groups = len(groups)
            self.groups = groups

    def predict(self, X, y):
        self._check_fit()
        # Parallelize the computation of the importance scores for each group
        out_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_predict_one_group)(X, y, index_j, j)
            for index_j, j in enumerate(self.groups.keys())
        )
        return np.stack(out_list, axis=0)

    def score(self, X, y):
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

    def _check_fit(self):
        pass

    def _joblib_predict_one_group(self, X, y, index_j, j):
        if isinstance(X, pd.DataFrame):
            X_j = X[self.groups[j]].copy().values
            X_minus_j = X.drop(columns=self.groups[j]).values
            group_ids = [i for i, col in enumerate(X.columns) if col in self.groups[j]]
            non_group_ids = [
                i for i, col in enumerate(X.columns) if col not in self.groups[j]
            ]
        else:
            X_j = X[:, self.groups[j]].copy()
            X_minus_j = np.delete(X, self.groups[j], axis=1)
            group_ids = self.groups[j]
            non_group_ids = np.delete(np.arange(X.shape[1]), group_ids)

        X_perm_j = self._permutation(X_minus_j, X_j, group_ids, non_group_ids, index_j)
        # Reshape X_perm_j to allow for batch prediction
        X_perm_batch = X_perm_j.reshape(-1, X.shape[1])
        if isinstance(X, pd.DataFrame):
            X_perm_batch = pd.DataFrame(
                X_perm_batch.reshape(-1, X.shape[1]), columns=X.columns
            )
        y_pred_perm = getattr(self.estimator, self.method)(X_perm_batch)

        # In case of classification, the output is a 2D array. Reshape accordingly
        if y_pred_perm.ndim == 1:
            y_pred_perm = y_pred_perm.reshape(self.n_permutations, X.shape[0])
        else:
            y_pred_perm = y_pred_perm.reshape(
                self.n_permutations, X.shape[0], y_pred_perm.shape[1]
            )
        return y_pred_perm

    def _permutation(self, X_minus_j, X_j, group_ids, non_group_ids):
        raise NotImplementedError


class PermutationImportance(BasePermutation):
    def __init__(self, *arg, random_state: int = None, **kwarg):
        super().__init__(*arg, **kwarg)
        self.rng = np.random.RandomState(random_state)

    def _permutation(self, X_minus_j, X_j, group_ids, non_group_ids, index_j):
        # Create an array X_perm_j of shape (n_permutations, n_samples, n_features)
        # where the j-th group of covariates is permuted
        X_perm_j = np.empty(
            (self.n_permutations, X_minus_j.shape[0], X_minus_j.shape[1] + X_j.shape[1])
        )
        X_perm_j[:, :, non_group_ids] = X_minus_j
        # Create the permuted data for the j-th group of covariates
        group_j_permuted = np.array(
            [self.rng.permutation(X_j) for _ in range(self.n_permutations)]
        )
        X_perm_j[:, :, group_ids] = group_j_permuted
        return X_perm_j


class LOCO(BasePermutation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, n_permutations=1, **kwargs)
        self._list_estimators = []

    def fit(self, X, y, groups=None):
        super().fit(X, y, groups)
        # create a list of covariate estimators for each group if not provided
        self._list_estimators = [clone(self.estimator) for _ in range(self.n_groups)]

        # Parallelize the fitting of the covariate estimators
        self._list_estimators = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_fit_one_group)(estimator, X, y, j)
            for j, estimator in zip(self.groups.keys(), self._list_estimators)
        )
        return self

    def _joblib_fit_one_group(self, estimator, X, y, j):
        if isinstance(X, pd.DataFrame):
            X_minus_j = X.drop(columns=self.groups[j])
        else:
            X_minus_j = np.delete(X, self.groups[j], axis=1)
        estimator.fit(X_minus_j, y)
        return estimator

    def _joblib_predict_one_group(self, X, y, index_j, j):
        if isinstance(X, pd.DataFrame):
            X_minus_j = X.drop(columns=self.groups[j])
        else:
            X_minus_j = np.delete(X, self.groups[j], axis=1)

        y_pred_loco = getattr(self._list_estimators[index_j], self.method)(X_minus_j)

        return [y_pred_loco]

    def _check_fit(self):
        check_is_fitted(self.estimator)
        if len(self._list_estimators) == 0:
            raise ValueError("The estimators require to be fit before to use them")
        for m in self._list_estimators:
            check_is_fitted(m)


class CPI(BasePermutation):
    def __init__(self, *arg, random_state: int = None, **kwarg):
        super().__init__(*arg, **kwarg)
        self.rng = np.random.RandomState(random_state)
        self._list_imputation_models = []

    def fit(self, X, imputation_model, groups=None):
        super().fit(X, None, groups)
        if isinstance(imputation_model, list):
            self._list_imputation_models = imputation_model
        else:
            self._list_imputation_models = [
                clone(imputation_model) for _ in range(self.n_groups)
            ]

        # Parallelize the fitting of the covariate estimators
        self._list_imputation_models = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_fit_one_group)(estimator, X, j)
            for j, estimator in zip(self.groups.keys(), self._list_imputation_models)
        )

        return self

    def _joblib_fit_one_group(self, estimator, X, j):
        if isinstance(X, pd.DataFrame):
            X_j = X[self.groups[j]].copy().values
            X_minus_j = X.drop(columns=self.groups[j]).values
        else:
            X_j = X[:, self.groups[j]].copy()
            X_minus_j = np.delete(X, self.groups[j], axis=1)
        estimator.fit(X_minus_j, X_j)
        return estimator

    def _check_fit(self):
        if len(self._list_imputation_models) == 0:
            raise ValueError("The estimators require to be fit before to use them")
        for m in self._list_imputation_models:
            check_is_fitted(m)

    def _permutation(self, X_minus_j, X_j, group_ids, non_group_ids, index_j):
        X_j_hat = (
            self._list_imputation_models[index_j].predict(X_minus_j).reshape(X_j.shape)
        )
        residual_j = X_j - X_j_hat

        # Create an array X_perm_j of shape (n_permutations, n_samples, n_features)
        # where the j-th group of covariates is (conditionally) permuted
        X_perm_j = np.empty(
            (self.n_permutations, X_minus_j.shape[0], X_minus_j.shape[1] + X_j.shape[1])
        )
        X_perm_j[:, :, non_group_ids] = X_minus_j
        # Create the permuted data for the j-th group of covariates
        residual_j_perm = np.array(
            [self.rng.permutation(residual_j) for _ in range(self.n_permutations)]
        )
        X_perm_j[:, :, group_ids] = X_j_hat[np.newaxis, :, :] + residual_j_perm
        return X_perm_j
