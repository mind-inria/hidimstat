import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import check_is_fitted, clone
from sklearn.metrics import mean_squared_error

from hidimstat._utils.docstring import _aggregate_docstring
from hidimstat._utils.utils import (
    _get_array_cols,
    check_scoring,
    check_statistical_test,
)
from hidimstat.base_perturbation import BasePerturbation, BasePerturbationCV


class LOCO(BasePerturbation):
    """
    Leave-One-Covariate-Out (LOCO) algorithm

    This method is presented in :footcite:t:`lei2018distribution` and :footcite:t:`verdinelli2024feature`.
    The model is re-fitted for each feature/group of features. The importance is
    then computed as the difference between the loss of the full model and the loss
    of the model without the feature/group.

    Parameters
    ----------
    estimator : sklearn compatible estimator
        The estimator to use for the prediction.
    scoring : srt, callable
        Strategy to evaluate the performance of the estimator to compute
        importance scores. Based on :func:`sklearn.metrics.check_scoring`.
    statistical_test : callable or str, default="ttest"
        Statistical test function for computing p-values of importance scores.
    feature_groups: dict or None, default=None
        A dictionary where the keys are the group names and the values are the
        list of column names corresponding to each features group. If None,
        the feature_groups are identified based on the columns of X.
    n_jobs : int, default=1
        The number of jobs to run in parallel. Parallelization is done over the
        variables or groups of variables.

    Notes
    -----
    :footcite:t:`Williamson_General_2023` also presented a LOCO method with an
    additional data splitting strategy.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        estimator,
        scoring=None,
        statistical_test="ttest",
        feature_groups=None,
        n_jobs: int = 1,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_permutations=1,
            statistical_test=statistical_test,
            feature_groups=feature_groups,
            n_jobs=n_jobs,
        )
        # internal variable
        self._list_estimators = None

    def fit(self, X, y):
        """
        Fit a model after removing each covariate/group of covariates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        super().fit(X, y)
        # create a list of covariate estimators for each group if not provided
        self._list_estimators = [
            clone(self.estimator) for _ in range(self.n_feature_groups_)
        ]

        # Parallelize the fitting of the covariate estimators
        self._list_estimators = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_fit_one_features_group)(
                estimator, X, y, feature_groups_ids
            )
            for feature_groups_ids, estimator in zip(
                self._feature_groups_ids,
                self._list_estimators,
                strict=False,
            )
        )
        return self

    def _joblib_fit_one_features_group(
        self, estimator, X, y, feature_groups_ids
    ):
        """Fit the estimator after removing a group of covariates. Used in parallel."""
        X_minus_j = _get_array_cols(X, feature_groups_ids, drop=True)
        estimator.fit(X_minus_j, y)
        return estimator

    def _joblib_score_one_feature_group(
        self, X, y, features_group_id, random_state=None
    ):
        """Predict the target feature after removing a group of covariates.
        Used in parallel.
        """
        del random_state  # not used (only there for API compatibility)
        # Since we don't have access to column names, we use the member _feature_groups_ids
        X_minus_j = _get_array_cols(
            X, self._feature_groups_ids[features_group_id], drop=True
        )

        scoring_loco = self.scoring(
            self._list_estimators[features_group_id], X_minus_j, y
        )

        return [scoring_loco]

    def _check_fit(self):
        """Check that an estimator has been fitted after removing each group of
        covariates.
        """
        super()._check_fit()
        check_is_fitted(self.estimator)
        if self._list_estimators is None:
            raise ValueError(
                "The estimators require to be fit before to use them"
            )
        for m in self._list_estimators:
            check_is_fitted(m)


def loco_importance(
    estimator,
    X,
    y,
    scoring=None,
    feature_groups=None,
    test_statistic="ttest",
    k_best=None,
    percentile=None,
    threshold_min=None,
    threshold_max=None,
    n_jobs: int = 1,
):
    warnings.warn(
        "loco_importance is deprecated and will be removed in version 0.6.0. "
        "Please use class LOCO instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    method = LOCO(
        estimator=estimator,
        scoring=scoring,
        statistical_test=test_statistic,
        feature_groups=feature_groups,
        n_jobs=n_jobs,
    )
    method.fit_importance(X, y)
    selection = method.importance_selection(
        k_best=k_best,
        percentile=percentile,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
    )
    return selection, method.importances_, method.pvalues_


# use the docstring of the class for the function
loco_importance.__doc__ = _aggregate_docstring(
    [
        LOCO.__doc__,
        LOCO.__init__.__doc__,
        LOCO.fit_importance.__doc__,
        LOCO.importance_selection.__doc__,
    ],
    """
Returns
-------
selection : ndarray of shape (n_groups,)
    Boolean array indicating selected feature groups (True = selected).
importances : ndarray of shape (n_groups,)
    Feature group importance scores/test statistics.
pvalues : ndarray of shape (n_groups,)
    None because there is no p-value for this method.
""",
)


class LOCOCV(BasePerturbationCV):
    """
    Leave-One-Covariate-Out (LOCO) algorithm with Cross-Validation.

    Parameters
    ----------
    estimators: list of sklearn estimators or single sklearn estimator
        Can be a list of fitted sklearn estimators (one per fold) or a single sklearn
        estimator that will then be cloned and fitted on each fold.
    cv: cross-validation generator
        A cross-validation generator object (e.g., KFold, StratifiedKFold).
    statistical_test : callable or str, default="nb-ttest"
        Statistical test function for computing p-values from importance scores.
    scoring : srt, callable
        Strategy to evaluate the performance of the estimator to compute
        importance scores. Based on :func:`sklearn.metrics.check_scoring`.
    feature_groups: dict or None, default=None
        A dictionary where the keys are the group names and the values are the
        list of column names corresponding to each features group. If None,
        the feature_groups are identified based on the columns of X.
    n_jobs : int, default=1
        The number of jobs to run in parallel. Parallelization is done over the folds.

    Attributes
    ----------
    importance_estimators_ : list of LOCO instances
        The LOCO instances fitted on each fold.
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
        scoring=None,
        statistical_test="nb-ttest",
        feature_groups=None,
        n_jobs=1,
    ):
        super().__init__(estimators, cv, statistical_test, n_jobs)
        self.scoring = scoring
        self.feature_groups = feature_groups

    def _fit_single_split(self, estimator, X_train, y_train):
        """Fit a LOCO instance on a single train/test split."""
        loco = LOCO(
            estimator=estimator,
            scoring=self.scoring,
            feature_groups=self.feature_groups,
            n_jobs=1,  # no parallelization inside the fold
        )
        loco.fit(X_train, y_train)
        return loco
