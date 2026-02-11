import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, check_is_fitted, clone
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import mean_squared_error

from hidimstat._utils.docstring import _aggregate_docstring
from hidimstat.base_perturbation import BasePerturbation, BasePerturbationCV
from hidimstat.samplers.conditional_sampling import ConditionalSampler


class CFI(BasePerturbation):
    """
    Conditional Feature Importance (CFI) algorithm.
    :footcite:t:`Chamma_NeurIPS2023` and for group-level see
    :footcite:t:`Chamma_AAAI2024`.

    Parameters
    ----------
    estimator : sklearn compatible estimator
        The estimator to use for the prediction.
    method : str, default="predict"
        The method to use for the prediction. This determines the predictions passed
        to the loss function. Supported methods are "predict", "predict_proba" or
        "decision_function".
    loss : callable, default=mean_squared_error
        The loss function to use when comparing the perturbed model to the full
        model.
    n_permutations : int, default=50
        The number of permutations to perform. For each variable/group of variables,
        the mean of the losses over the `n_permutations` is computed.
    imputation_model_continuous : sklearn compatible estimator, default=RidgeCV()
        The model used to estimate the conditional distribution of a given
        continuous variable/group of variables given the others.
    imputation_model_categorical : sklearn compatible estimator, default=LogisticRegressionCV()
        The model used to estimate the conditional distribution of a given
        categorical variable/group of variables given the others. Binary is
        considered as a special case of categorical.
    features_groups: dict or None, default=None
        A dictionary where the keys are the group names and the values are the
        list of column names corresponding to each features group. If None,
        the features_groups are identified based on the columns of X.
    feature_types: str or list, default="auto"
        The feature type. Supported types include "auto", "continuous", and
        "categorical". If "auto", the type is inferred from the cardinality
        of the unique values passed to the `fit` method.
    categorical_max_cardinality : int, default=10
        The maximum cardinality of a variable to be considered as categorical
        when the variable type is inferred (set to "auto" or not provided).
    statistical_test : callable or str, default="ttest"
        Statistical test function for computing p-values of importance scores.
    random_state : int or None, default=None
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
        loss: callable = mean_squared_error,
        n_permutations: int = 50,
        imputation_model_continuous=RidgeCV(),
        imputation_model_categorical=LogisticRegressionCV(),
        features_groups=None,
        feature_types="auto",
        categorical_max_cardinality: int = 10,
        statistical_test="ttest",
        random_state: int | None = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            estimator=estimator,
            method=method,
            loss=loss,
            n_permutations=n_permutations,
            statistical_test=statistical_test,
            n_jobs=n_jobs,
            features_groups=features_groups,
            random_state=random_state,
        )

        # check the validity of the inputs
        assert imputation_model_continuous is None or issubclass(
            imputation_model_continuous.__class__, BaseEstimator
        ), "Continuous imputation model invalid"
        assert imputation_model_categorical is None or issubclass(
            imputation_model_categorical.__class__, BaseEstimator
        ), "Categorial imputation model invalid"

        self.feature_types = feature_types
        self._list_imputation_models = []
        self.categorical_max_cardinality = categorical_max_cardinality
        self.imputation_model_categorical = imputation_model_categorical
        self.imputation_model_continuous = imputation_model_continuous

    def fit(self, X, y=None):
        """
        Fit the imputation models.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        y: array-like of shape (n_samples,)
            Not used, only present for consistency with the sklearn API.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        del y
        super().fit(X, None)

        # check the feature type
        if isinstance(self.feature_types, str):
            if self.feature_types in ["auto", "continuous", "categorical"]:
                self.feature_types = [
                    self.feature_types for _ in range(self.n_features_groups_)
                ]
            else:
                raise ValueError(
                    "feature_types support only the string 'auto', 'continuous', 'categorical'"
                )

        self._list_imputation_models = [
            ConditionalSampler(
                data_type=self.feature_types[features_group_id],
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
                categorical_max_cardinality=self.categorical_max_cardinality,
            )
            for features_group_id in range(self.n_features_groups_)
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

    def fit_importance(self, X, y):
        """
        Fits the model to the data and computes feature importance scores.
        Convenience method that combines fit() and importance() into a single call.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        importances_ : ndarray of shape (n_groups,)
            The calculated importance scores for each feature group.
            Higher values indicate greater importance.

        Notes
        -----
        This method first calls fit() to identify feature groups, then calls
        importance() to compute the importance scores for each group.
        """
        self.fit(X, y)
        return self.importance(X, y)

    def _joblib_fit_one_features_group(
        self, estimator, X, features_groups_ids
    ):
        """Fit a single imputation model, for a single group of features. This method
        is parallelized.
        """
        X_j = X[:, features_groups_ids].copy()
        X_minus_j = np.delete(X, features_groups_ids, axis=1)
        estimator.fit(X_minus_j, X_j)
        return estimator

    def _check_fit(self):
        """
        Check if base class and imputation models have been fitted.

        Raises
        ------
        ValueError
            If the class has not been fitted (i.e., if n_features_groups_
            or _features_groups_ids attributes are missing).
            If the class has not been fitted or imputation models are not fitted.

        """
        super()._check_fit()
        if len(self._list_imputation_models) == 0:
            raise ValueError(
                "The imputation models require to be fitted before being used."
            )
        for m in self._list_imputation_models:
            check_is_fitted(m.model)

    def _permutation(self, X, features_group_id, random_state=None):
        """Sample from the conditional distribution using a permutation of the
        residuals.
        """
        X_j = X[:, self._features_groups_ids[features_group_id]].copy()
        X_minus_j = np.delete(
            X, self._features_groups_ids[features_group_id], axis=1
        )
        return self._list_imputation_models[features_group_id].sample(
            X_minus_j,
            X_j,
            n_samples=self.n_permutations,
            random_state=random_state,
        )


def cfi_importance(
    estimator,
    X,
    y,
    method: str = "predict",
    loss: callable = mean_squared_error,
    n_permutations: int = 50,
    imputation_model_continuous=RidgeCV(),
    imputation_model_categorical=LogisticRegressionCV(),
    features_groups=None,
    feature_types="auto",
    categorical_max_cardinality: int = 10,
    test_statistic="ttest",
    k_best=None,
    percentile=None,
    threshold_max=None,
    threshold_min=None,
    random_state: int | None = None,
    n_jobs: int = 1,
):
    methods = CFI(
        estimator=estimator,
        method=method,
        loss=loss,
        n_permutations=n_permutations,
        imputation_model_continuous=imputation_model_continuous,
        imputation_model_categorical=imputation_model_categorical,
        features_groups=features_groups,
        feature_types=feature_types,
        categorical_max_cardinality=categorical_max_cardinality,
        statistical_test=test_statistic,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    methods.fit_importance(X, y)
    selection = methods.importance_selection(
        k_best=k_best,
        percentile=percentile,
        threshold_max=threshold_max,
        threshold_min=threshold_min,
    )
    return selection, methods.importances_, methods.pvalues_


# use the docstring of the class for the function
cfi_importance.__doc__ = _aggregate_docstring(
    [
        CFI.__doc__,
        CFI.__init__.__doc__,
        CFI.fit_importance.__doc__,
        CFI.importance_selection.__doc__,
    ],
    """
    Returns
    -------
    selection : ndarray of shape (n_features,)
        Boolean array indicating selected features (True = selected)
    importances : ndarray of shape (n_features,)
        Feature importance scores/test statistics.
    pvalues : ndarray of shape (n_features,)
        P-values for importance scores.
    """,
)


class CFICV(BasePerturbationCV):
    """
    Conditional Feature Importance (CFI) algorithm with Cross-Validation.

    Parameters
    ----------
    estimators: list of sklearn estimators or single sklearn estimator
        Can be a list of fitted sklearn estimators (one per fold) or a single sklearn
        estimator that will then be cloned and fitted on each fold.
    cv: cross-validation generator
        A cross-validation generator object (e.g., KFold, StratifiedKFold).
    statistical_test : callable or str, default="nb-ttest"
        Statistical test function for computing p-values from importance scores.
    method : str, default="predict"
        The method to use for the prediction. This determines the predictions passed
        to the loss function. Supported methods are "predict", "predict_proba" or
        "decision_function".
    loss : callable, default=mean_squared_error
        The loss function to use when comparing the perturbed model to the full
        model.
    n_permutations : int, default=50
        The number of permutations to perform. For each variable/group of variables,
        the mean of the losses over the `n_permutations` is computed.
    imputation_model_continuous : sklearn compatible estimator, default=RidgeCV()
        The model used to estimate the conditional distribution of a given
        continuous variable/group of variables given the others.
    imputation_model_categorical : sklearn compatible estimator, default=LogisticRegressionCV()
        The model used to estimate the conditional distribution of a given
        categorical variable/group of variables given the others. Binary is
        considered as a special case of categorical.
    features_groups: dict or None, default=None
        A dictionary where the keys are the group names and the values are the
        list of column names corresponding to each features group. If None,
        the features_groups are identified based on the columns of X.
    feature_types: str or list, default="auto"
        The feature type. Supported types include "auto", "continuous", and
        "categorical". If "auto", the type is inferred from the cardinality
        of the unique values passed to the `fit` method.
    categorical_max_cardinality : int, default=10
        The maximum cardinality of a variable to be considered as categorical
        when the variable type is inferred (set to "auto" or not provided).s.
    random_state : int or None, default=None
        The random state to use for sampling.
    n_jobs : int, default=1
        The number of jobs to run in parallel. Parallelization is done over the folds.

    Attributes
    ----------
    importance_estimators_ : list of CFI instances
        The CFI instances fitted on each fold.
    importances_ : ndarray of shape (n_groups, n_folds)
        The calculated importance scores for each feature group and each fold.
        Higher values indicate greater importance.
    pvalues_ : ndarray of shape (n_groups,)
        The p-values for the importance scores computed across folds.
    estimators_ : list of sklearn estimators
        List of fitted estimators for each fold.
    test_train_frac_ : float
        Fraction of test samples over train samples in each fold. Approximated as
        1 / (n_splits - 1).
    """

    def __init__(
        self,
        estimators,
        cv,
        statistical_test="nb-ttest",
        method="predict",
        loss=mean_squared_error,
        n_permutations=50,
        imputation_model_continuous=RidgeCV(),
        imputation_model_categorical=LogisticRegressionCV(),
        features_groups=None,
        feature_types="auto",
        categorical_max_cardinality=10,
        random_state=None,
        n_jobs=1,
    ):
        super().__init__(estimators, cv, statistical_test, n_jobs)
        self.method = method
        self.loss = loss
        self.n_permutations = n_permutations
        self.imputation_model_continuous = imputation_model_continuous
        self.imputation_model_categorical = imputation_model_categorical
        self.features_groups = features_groups
        self.feature_types = feature_types
        self.categorical_max_cardinality = categorical_max_cardinality
        self.random_state = random_state

    def _fit_single_split(self, estimator, X_train, y_train):
        """Fit a CFI instance on a single train/test split."""
        cfi = CFI(
            estimator=estimator,
            method=self.method,
            loss=self.loss,
            n_permutations=self.n_permutations,
            imputation_model_continuous=self.imputation_model_continuous,
            imputation_model_categorical=self.imputation_model_categorical,
            features_groups=self.features_groups,
            feature_types=self.feature_types,
            categorical_max_cardinality=self.categorical_max_cardinality,
            random_state=self.random_state,
            n_jobs=1,  # no parallelization inside the fold
        )
        cfi.fit(X_train, y_train)
        return cfi
