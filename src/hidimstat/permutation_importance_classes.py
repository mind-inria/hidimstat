import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, check_is_fitted, clone
from sklearn.metrics import root_mean_squared_error

from hidimstat.conditional_sampling import ConditionalSampler
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
        Base class for model agnostic variable importance measures based on
        perturbation.

        Parameters
        ----------
        estimator : object
            The estimator to use for the prediction.
        loss : callable, default=root_mean_squared_error
            The loss function to use when comparing the perturbed model to the full
            model.
        n_permutations : int, default=50
            Only for PermutationImportance or CPI. The number of permutations to
            perform.
        method : str, default="predict"
            The method to use for the prediction. This determines the predictions passed
            to the loss function.
        n_jobs : int, default=1
            The number of jobs to run in parallel. Parallelization is done over the
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


class PermutationImportance(BasePerturbation):
    def __init__(
        self,
        estimator,
        loss: callable = root_mean_squared_error,
        method: str = "predict",
        n_jobs: int = 1,
        n_permutations: int = 50,
        random_state: int = None,
    ):
        """
        Permutation Importance algorithm as presented in
        :footcite:t:`breimanRandomForests2001`. For each variable/group of variables,
        the importance is computed as the difference between the loss of the initial
        model and the loss of the model with the variable/group permuted.

        Parameters
        ----------
        estimator : object
            The estimator to use for the prediction.
        loss : callable, default=root_mean_squared_error
            The loss function to use when comparing the perturbed model to the full
            model.
        method : str, default="predict"
            The method to use for the prediction. This determines the predictions passed
            to the loss function.
        n_jobs : int, default=1
            The number of jobs to run in parallel. Parallelization is done over the
            variables or groups of variables.
        n_permutations : int, default=50
            The number of permutations to perform. For each variable/group of variables,
            the mean of the losses over the `n_permutations` is computed.
        random_state : int, default=None
            The random state to use for sampling.
        """
        super().__init__(
            estimator=estimator,
            loss=loss,
            method=method,
            n_jobs=n_jobs,
            n_permutations=n_permutations,
        )
        self.rng = np.random.RandomState(random_state)

    def _permutation(self, X, group_id):
        # Create the permuted data for the j-th group of covariates
        X_perm_j = np.array(
            [
                self.rng.permutation(X[:, self._groups_ids[group_id]].copy())
                for _ in range(self.n_permutations)
            ]
        )
        return X_perm_j


class LOCO(BasePerturbation):
    def __init__(
        self,
        estimator,
        loss: callable = root_mean_squared_error,
        method: str = "predict",
        n_jobs: int = 1,
    ):
        """
        Leave-One-Covariate-Out (LOCO) as presented in
        :footcite:t:`Williamson_General_2023`. The model is re-fitted for each variable/
        group of variables. The importance is then computed as the difference between
        the loss of the full model and the loss of the model without the variable/group.

        Parameters
        ----------
        estimator : object
            The estimator to use for the prediction.
        loss : callable, default=root_mean_squared_error
            The loss function to use when comparing the perturbed model to the full
            model.
        method : str, default="predict"
            The method to use for the prediction. This determines the predictions passed
            to the loss function.
        n_jobs : int, default=1
            The number of jobs to run in parallel. Parallelization is done over the
            variables or groups of variables.
        """
        super().__init__(
            estimator=estimator,
            loss=loss,
            method=method,
            n_jobs=n_jobs,
            n_permutations=1,
        )
        self._list_estimators = []

    def fit(self, X, y, groups=None):
        super().fit(X, y, groups)
        # create a list of covariate estimators for each group if not provided
        self._list_estimators = [clone(self.estimator) for _ in range(self.n_groups)]

        # Parallelize the fitting of the covariate estimators
        self._list_estimators = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_fit_one_group)(estimator, X, y, key_groups)
            for key_groups, estimator in zip(self.groups.keys(), self._list_estimators)
        )
        return self

    def _joblib_fit_one_group(self, estimator, X, y, key_groups):
        if isinstance(X, pd.DataFrame):
            X_minus_j = X.drop(columns=self.groups[key_groups])
        else:
            X_minus_j = np.delete(X, self.groups[key_groups], axis=1)
        estimator.fit(X_minus_j, y)
        return estimator

    def _joblib_predict_one_group(self, X, group_id, key_groups):
        X_minus_j = np.delete(X, self._groups_ids[group_id], axis=1)

        y_pred_loco = getattr(self._list_estimators[group_id], self.method)(X_minus_j)

        return [y_pred_loco]

    def _check_fit(self):
        check_is_fitted(self.estimator)
        if len(self._list_estimators) == 0:
            raise ValueError("The estimators require to be fit before to use them")
        for m in self._list_estimators:
            check_is_fitted(m)


class CPI(BasePerturbation):

    def __init__(
        self,
        estimator,
        loss: callable = root_mean_squared_error,
        method: str = "predict",
        n_jobs: int = 1,
        n_permutations: int = 50,
        imputation_model_continuous=None,
        imputation_model_binary=None,
        imputation_model_classification=None,
        random_state: int = None,
        categorical_max_cardinality: int = 10,
    ):
        """
        Conditional Permutation Importance (CPI) algorithm.
        :footcite:t:`Chamma_NeurIPS2023` and for group-level see
        :footcite:t:`Chamma_AAAI2024`.

        Parameters
        ----------
        estimator : object
            The estimator to use for the prediction.
        loss : callable, default=root_mean_squared_error
            The loss function to use when comparing the perturbed model to the full
            model.
        method : str, default="predict"
            The method to use for the prediction. This determines the predictions passed
            to the loss function.
        n_jobs : int, default=1
            The number of jobs to run in parallel. Parallelization is done over the
            variables or groups of variables.
        n_permutations : int, default=50
            The number of permutations to perform. For each variable/group of variables,
            the mean of the losses over the `n_permutations` is computed.
        imputation_model_continuous : object, default=None
            The model used to estimate the conditional distribution of a given
            continuous variable/group of variables given the others.
        imputation_model_binary : object, default=None
            The model used to estimate the conditional distribution of a given
            binary variable/group of variables given the others.
        imputation_model_classification : object, default=None
            The model used to estimate the conditional distribution of a given
            categorical variable/group of variables given the others.
        random_state : int, default=None
            The random state to use for sampling.
        categorical_max_cardinality : int, default=10
            The maximum cardinality of a variable to be considered as categorical
            when the variable type is inferred (set to "auto" or not provided).
        """

        super().__init__(
            estimator=estimator,
            loss=loss,
            method=method,
            n_jobs=n_jobs,
            n_permutations=n_permutations,
        )
        self.rng = np.random.RandomState(random_state)
        self._list_imputation_models = []

        self.imputation_model = {
            "continuous": imputation_model_continuous,
            "binary": imputation_model_binary,
            "categorical": imputation_model_classification,
        }
        self.categorical_max_cardinality = categorical_max_cardinality

    def fit(self, X, groups=None, var_type="auto"):
        super().fit(X, None, groups=groups)
        if isinstance(var_type, str):
            self.var_type = [var_type for _ in range(self.n_groups)]
        else:
            self.var_type = var_type

        self._list_imputation_models = [
            ConditionalSampler(
                data_type=self.var_type[groupd_id],
                model_regression=(
                    None
                    if self.imputation_model["continuous"] is None
                    else clone(self.imputation_model["continuous"])
                ),
                model_binary=(
                    None
                    if self.imputation_model["binary"] is None
                    else clone(self.imputation_model["binary"])
                ),
                model_categorical=(
                    None
                    if self.imputation_model["categorical"] is None
                    else clone(self.imputation_model["categorical"])
                ),
                random_state=self.rng,
                categorical_max_cardinality=self.categorical_max_cardinality,
            )
            for groupd_id in range(self.n_groups)
        ]

        # Parallelize the fitting of the covariate estimators
        X_ = np.asarray(X)
        self._list_imputation_models = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_fit_one_group)(estimator, X_, groups_ids)
            for groups_ids, estimator in zip(
                self._groups_ids, self._list_imputation_models
            )
        )

        return self

    def _joblib_fit_one_group(self, estimator, X, groups_ids):
        X_ = self._remove_nan(X)
        X_j = X_[:, groups_ids].copy()
        X_minus_j = np.delete(X_, groups_ids, axis=1)
        estimator.fit(X_minus_j, X_j)
        return estimator

    def _check_fit(self):
        if len(self._list_imputation_models) == 0:
            raise ValueError("The estimators require to be fit before to use them")
        for m in self._list_imputation_models:
            check_is_fitted(m.model)

    def _permutation(self, X, group_id):
        X_ = self._remove_nan(X)
        X_j = X_[:, self._groups_ids[group_id]].copy()
        X_minus_j = np.delete(X_, self._groups_ids[group_id], axis=1)
        return self._list_imputation_models[group_id].sample(
            X_minus_j, X_j, n_samples=self.n_permutations
        )

    def _remove_nan(self, X):
        # TODO: specify the strategy to handle NaN values
        return X
