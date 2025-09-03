import numpy as np
from joblib import Parallel, delayed
from sklearn.base import check_is_fitted, clone, BaseEstimator
from sklearn.metrics import root_mean_squared_error
from sklearn.utils.validation import check_random_state

from hidimstat.base_perturbation import BasePerturbation
from hidimstat.conditional_sampling import ConditionalSampler


class CFI(BasePerturbation):
    def __init__(
        self,
        estimator,
        loss: callable = root_mean_squared_error,
        method: str = "predict",
        n_jobs: int = 1,
        n_permutations: int = 50,
        imputation_model_continuous=None,
        imputation_model_categorical=None,
        random_state: int = None,
        categorical_max_cardinality: int = 10,
    ):
        """
        Conditional Feature Importance (CFI) algorithm.
        :footcite:t:`Chamma_NeurIPS2023` and for group-level see
        :footcite:t:`Chamma_AAAI2024`.

        Parameters
        ----------
        estimator : sklearn compatible estimator, optional
            The estimator to use for the prediction.
        loss : callable, default=root_mean_squared_error
            The loss function to use when comparing the perturbed model to the full
            model.
        method : str, default="predict"
            The method to use for the prediction. This determines the predictions passed
            to the loss function. Supported methods are "predict", "predict_proba",
            "decision_function", "transform".
        n_jobs : int, default=1
            The number of jobs to run in parallel. Parallelization is done over the
            features or groups of features.
        n_permutations : int, default=50
            The number of permutations to perform. For each feature/group of features,
            the mean of the losses over the `n_permutations` is computed.
        imputation_model_continuous : sklearn compatible estimator, optional
            The model used to estimate the conditional distribution of a given
            continuous features/group of features given the others.
        imputation_model_categorical : sklearn compatible estimator, optional
            The model used to estimate the conditional distribution of a given
            categorical features/group of features given the others. Binary is
            considered as a special case of categorical.
        random_state : int, default=None
            The random state to use for sampling.
        categorical_max_cardinality : int, default=10
            The maximum cardinality of a feature to be considered as categorical
            when the feature type is inferred (set to "auto" or not provided).

        References
        ----------
        .. footbibliography::
        """
        super().__init__(
            estimator=estimator,
            loss=loss,
            method=method,
            n_jobs=n_jobs,
            n_permutations=n_permutations,
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

    def fit(self, X, y=None, features_groups=None, features_type="auto"):
        """Fit the imputation models.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        y: array-like of shape (n_samples,)
            Not used, only present for consistency with the sklearn API.
        features_groups: dict, optional
            A dictionary where the keys are the group names and the values are the
            list of column names corresponding to each features group. If None,
            the features_groups are identified based on the columns of X.
        features_type: str or list, default="auto"
            The feature type. Supported types include "auto", "continuous", and
            "categorical". If "auto", the type is inferred from the cardinality
            of the unique values passed to the `fit` method.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.random_state = check_random_state(self.random_state)
        super().fit(X, None, features_groups=features_groups)
        if isinstance(features_type, str):
            self.features_type = [features_type for _ in range(self.n_features_groups)]
        else:
            self.features_type = features_type

        self._list_imputation_models = [
            ConditionalSampler(
                data_type=self.features_type[features_groupd_id],
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
            for features_groupd_id in range(self.n_features_groups)
        ]

        # Parallelize the fitting of the covariate estimators
        X_ = np.asarray(X)
        self._list_imputation_models = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_fit_one_features_group)(
                imputation_model, X_, features_groups_ids
            )
            for features_groups_ids, imputation_model in zip(
                self._features_groups_ids, self._list_imputation_models
            )
        )

        return self

    def _joblib_fit_one_features_group(self, estimator, X, features_groups_ids):
        """Fit a single imputation model, for a single group of features. This method
        is parallelized."""
        X_j = X[:, features_groups_ids].copy()
        X_minus_j = np.delete(X, features_groups_ids, axis=1)
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
            If the method has not been fitted (i.e., if n_features_groups, features_groups,
            or _features_groups_ids attributes are missing) or if imputation models
            are not fitted.
        AssertionError
            If the number of features in X does not match the total number
            of features in the grouped features.
        """
        super()._check_fit(X)
        if len(self._list_imputation_models) == 0:
            raise ValueError(
                "The imputation models require to be fitted before being used."
            )
        for m in self._list_imputation_models:
            check_is_fitted(m.model)

    def _permutation(self, X, features_group_id):
        """Sample from the conditional distribution using a permutation of the
        residuals."""
        X_j = X[:, self._features_groups_ids[features_group_id]].copy()
        X_minus_j = np.delete(X, self._features_groups_ids[features_group_id], axis=1)
        return self._list_imputation_models[features_group_id].sample(
            X_minus_j, X_j, n_samples=self.n_permutations
        )
