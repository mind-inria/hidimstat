import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, check_is_fitted, clone
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import mean_squared_error

from hidimstat._utils.docstring import _aggregate_docstring
from hidimstat._utils.utils import _get_array_cols
from hidimstat.base_perturbation import BasePerturbation, BasePerturbationCV
from hidimstat.samplers.conditional_sampling import ConditionalSampler


class HoldoutRandomTest(BasePerturbation):
    """
    Holdout Randomization Test (HRT) algorithm.

    The HRT is computed following the algorithm 2 of
    :footcite:t:`tansey2022holdout`. For each feature group, draw
    `n_permutations` samples from the conditional distribution, then compute
    the empirical risk using the `loss` function, that we denote as :math:`t_k`.
    The p-value is then computed as the proportion of :math:`t_k` that are greater than
    the empirical risk of the original data that we denote as :math:`t_0`. For
    a feature :math:`j` (resp. a feature group), the p-value is computed as:

    .. math::
        p_j = \\frac{1}{K+1} \\left(1 + \\sum_{k=1}^{K}
        \\mathbb{I}(t_k \\geq t_0)\\right)

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
        n_permutations: int = 100,
        imputation_model_continuous=None,
        imputation_model_categorical=None,
        features_groups=None,
        feature_types="auto",
        categorical_max_cardinality: int = 10,
        random_state: int | None = None,
        n_jobs: int = 1,
    ):

        self.estimator = estimator
        self.method = method
        self.loss = loss
        self.n_permutations = n_permutations
        self.imputation_model_continuous = imputation_model_continuous
        self.imputation_model_categorical = imputation_model_categorical
        self.features_groups = features_groups
        self.feature_types = feature_types
        self.categorical_max_cardinality = categorical_max_cardinality
        self.random_state = random_state
        self.n_jobs = n_jobs

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
        # check the validity of the inputs
        assert self.imputation_model_continuous is None or issubclass(
            self.imputation_model_continuous.__class__, BaseEstimator
        ), "Continuous imputation model invalid"
        assert self.imputation_model_categorical is None or issubclass(
            self.imputation_model_categorical.__class__, BaseEstimator
        ), "Categorial imputation model invalid"

        if self.imputation_model_continuous is None:
            self.imputation_model_continuous = RidgeCV()
        if self.imputation_model_categorical is None:
            self.imputation_model_categorical = LogisticRegressionCV(
                l1_ratios=(0,)
            )

        super().fit(X, y)

        # check the feature type
        if isinstance(self.feature_types, str):
            if self.feature_types in ["auto", "continuous", "categorical"]:
                self.feature_types_ = [
                    self.feature_types for _ in range(self.n_features_groups_)
                ]
            else:
                raise ValueError(
                    "feature_types support only the string 'auto', 'continuous', 'categorical'"
                )
        else:
            self.feature_types_ = self.feature_types

        self._list_imputation_models = [
            ConditionalSampler(
                data_type=self.feature_types_[features_group_id],
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
        self._list_imputation_models = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_fit_one_features_group)(
                imputation_model, X, features_groups_ids
            )
            for features_groups_ids, imputation_model in zip(
                self._features_groups_ids,
                self._list_imputation_models,
                strict=False,
            )
        )

        return self

    def importance(self, X, y):
        """
        Compute the importance of each feature group.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        y: array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        importance: array-like of shape (n_features_groups,)
            The importance of each feature group.
        """
