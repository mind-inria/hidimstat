import numpy as np
from joblib import Parallel, delayed
from sklearn.base import check_is_fitted, clone
from sklearn.metrics import root_mean_squared_error

from hidimstat.base_perturbation import BasePerturbation
from hidimstat.conditional_sampling import ConditionalSampler


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
