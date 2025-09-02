import numpy as np
from joblib import Parallel, delayed
from sklearn.base import check_is_fitted, clone, BaseEstimator
from sklearn.metrics import root_mean_squared_error
from sklearn.utils.validation import check_random_state

from hidimstat.base_perturbation import BasePerturbation
from hidimstat.conditional_sampling import ConditionalSampler


class CFI(BasePerturbation):
    """
    Conditional Feature Importance (CFI) algorithm.
    :footcite:t:`Chamma_NeurIPS2023` and for group-level see
    :footcite:t:`Chamma_AAAI2024`.

    Parameters
    ----------
    estimator : sklearn compatible estimator, optional
        The estimator to use for the prediction.
    method : str, default="predict"
        The method to use for the prediction. This determines the predictions passed
        to the loss function. Supported methods are "predict", "predict_proba",
        "decision_function", "transform".
    loss : callable, default=root_mean_squared_error
        The loss function to use when comparing the perturbed model to the full
        model.
    n_permutations : int, default=50
        The number of permutations to perform. For each variable/group of variables,
        the mean of the losses over the `n_permutations` is computed.
    imputation_model_continuous : sklearn compatible estimator, optional
        The model used to estimate the conditional distribution of a given
        continuous variable/group of variables given the others.
    imputation_model_categorical : sklearn compatible estimator, optional
        The model used to estimate the conditional distribution of a given
        categorical variable/group of variables given the others. Binary is
        considered as a special case of categorical.
    categorical_max_cardinality : int, default=10
        The maximum cardinality of a variable to be considered as categorical
        when the variable type is inferred (set to "auto" or not provided).
    random_state : int, default=None
        The random state to use for sampling.
    n_jobs : int, default=1
        The number of jobs to run in parallel. Parallelization is done over the
        variables or groups of variables.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        estimator,
        method: str = "predict",
        loss: callable = root_mean_squared_error,
        n_permutations: int = 50,
        imputation_model_continuous=None,
        imputation_model_categorical=None,
        categorical_max_cardinality: int = 10,
        random_state: int = None,
        n_jobs: int = 1,
    ):

        super().__init__(
            estimator=estimator,
            method=method,
            loss=loss,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
        )

        # check the validity of the inputs
        assert imputation_model_continuous is None or issubclass(
            imputation_model_continuous.__class__, BaseEstimator
        ), "Continous imputation model invalid"
        assert imputation_model_categorical is None or issubclass(
            imputation_model_categorical.__class__, BaseEstimator
        ), "Categorial imputation model invalid"

        self._list_imputation_models = []
        self.categorical_max_cardinality = categorical_max_cardinality
        self.imputation_model_categorical = imputation_model_categorical
        self.imputation_model_continuous = imputation_model_continuous
        self.random_state = random_state

    def fit(self, X, y=None, groups=None, var_type="auto"):
        """
        Fit the imputation models.

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
        var_type: str or list, default="auto"
            The variable type. Supported types include "auto", "continuous", and
            "categorical". If "auto", the type is inferred from the cardinality
            of the unique values passed to the `fit` method.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.random_state = check_random_state(self.random_state)
        super().fit(X, None, groups=groups)
        if isinstance(var_type, str):
            var_type = [var_type for _ in range(self._n_groups)]
        else:
            var_type = var_type

        self._list_imputation_models = [
            ConditionalSampler(
                data_type=var_type[groupd_id],
                model_regression=(
                    None
                    if self.imputation_model_continuous is None
                    else clone(self.imputation_model_continuous)
                ),
                model_categorical=(
                    None
                    if self.imputation_model_categorical is None
                    else clone(self.imputation_model_categorical)
                ),
                random_state=self.random_state,
                categorical_max_cardinality=self.categorical_max_cardinality,
            )
            for groupd_id in range(self._n_groups)
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
        """Fit a single imputation model, for a single group of variables. This method
        is parallelized."""
        X_j = X[:, groups_ids].copy()
        X_minus_j = np.delete(X, groups_ids, axis=1)
        estimator.fit(X_minus_j, X_j)
        return estimator

    def _check_fit(self, X):
        """
        Check if the perturbation method and imputation models have been properly fitted.

        This method verifies that both the base perturbation method and the imputation
        models have been fitted by checking required attributes and validating the input
        dimensions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to validate against the fitted model.

        Raises
        ------
        ValueError
            If the method has not been fitted (i.e., if n_groups, groups,
            or _groups_ids attributes are missing) or if imputation models
            are not fitted.
        AssertionError
            If the number of features in X does not match the total number
            of features in the grouped variables.
        """
        super()._check_fit(X)
        if len(self._list_imputation_models) == 0:
            raise ValueError(
                "The imputation models require to be fitted before being used."
            )
        for m in self._list_imputation_models:
            check_is_fitted(m.model)

    def _permutation(self, X, group_id):
        """Sample from the conditional distribution using a permutation of the
        residuals."""
        X_j = X[:, self._groups_ids[group_id]].copy()
        X_minus_j = np.delete(X, self._groups_ids[group_id], axis=1)
        return self._list_imputation_models[group_id].sample(
            X_minus_j, X_j, n_samples=self.n_permutations
        )
