import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import check_is_fitted, clone, is_classifier, is_regressor
from sklearn.dummy import DummyClassifier, DummyRegressor

from hidimstat._utils.docstring import _aggregate_docstring
from hidimstat._utils.utils import _get_array_cols
from hidimstat.base_perturbation import BasePerturbation, BasePerturbationCV


class LOCI(BasePerturbation):
    """
    Leave-One-Covariate-In (LOCI) algorithm

    The model is re-fitted on each single feature/group of features. The importance is
    then computed as the difference between the loss of an empty model (mean for regression,
    and majority vote for classification) and the loss of the model on the single feature/group.
    For more details, see :footcite:t:`ewald_2024`.

    Parameters
    ----------
    estimator : sklearn compatible estimator
        The estimator to use for the prediction.
    scoring : srt, callable
        Strategy to evaluate the performance of the estimator to compute
        importance scores. Based on :func:`sklearn.metrics.check_scoring`.
    method : str, default=None
        The method used for making predictions. This determines the predictions
        passed to the loss function. Supported methods are "predict",
        "predict_proba", "decision_function", "transform".

        .. deprecated:: 0.5.0
            Will be removed in 0.6.0. Please use parameter 'scoring' instead.
    loss : callable, default=None
        The function to compute the loss when comparing the perturbed model
        to the original model.

        .. deprecated:: 0.5.0
            Will be removed in 0.6.0. Please use parameter 'scoring' instead.
    statistical_test : callable or str, default="ttest"
        Statistical test function for computing p-values of importance scores.
    feature_groups: dict or None, default=None
        A dictionary where the keys are the group names and the values are the
        list of column names corresponding to each features group. If None,
        the feature_groups are identified based on the columns of X.
    n_jobs : int, default=1
        The number of jobs to run in parallel. Parallelization is done over the
        variables or groups of variables.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    .. versionadded:: 0.4.0
    """

    def __init__(
        self,
        estimator,
        scoring=None,
        method=None,
        loss=None,
        statistical_test="ttest",
        feature_groups=None,
        n_jobs: int = 1,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            method=method,
            loss=loss,
            statistical_test=statistical_test,
            feature_groups=feature_groups,
            n_jobs=n_jobs,
        )
        self._list_estimators = None

    def fit(self, X, y):
        """
        Fit a model for a single covariate/group of covariates.

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
        # Creating the baseline estimator for the scoring
        if is_classifier(self.estimator):
            self._dummy_estimator = DummyClassifier(strategy="prior")
        elif is_regressor(self.estimator):
            self._dummy_estimator = DummyRegressor(strategy="mean")
        else:
            raise TypeError(
                r"'self.estimator' must be a classifier or a regressor."
            )
        self._dummy_estimator_ = clone(self._dummy_estimator)
        self._dummy_estimator_.fit(X, y)
        return self

    def _joblib_fit_one_features_group(
        self, estimator, X, y, feature_groups_ids
    ):
        """
        Fit the estimator on a group of covariates.
        Used in parallel.
        """
        X_j = _get_array_cols(X, feature_groups_ids)
        estimator.fit(X_j, y)
        return estimator

    def _compute_loss_reference(self, X, y):
        """
        Compute the loss reference to which predictions from perturbed data
        will be compared.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.

        y: array-like of shape (n_samples,)
            The input groundtruth.

        Returns
        -------
        score: float
            The score of the underlying estimator on the data.
        """
        return self.scoring(self._dummy_estimator_, X, y)

    def _compute_score_difference(self):
        """
        Compute the loss difference between the reference loss
        and the loss computed from perturbed data.

        Returns
        -------
        score: array-like of shape (self.n_feature_groups_)
            The loss difference.
        """
        return np.array(
            [
                self.loss_reference_ - self.loss_[j]
                for j in range(self.n_feature_groups_)
            ]
        )

    def _joblib_score_one_feature_group(
        self, X, y, features_group_id, random_state=None
    ):
        """
        Predict the target feature for a single group of covariates.
        Used in parallel.
        """
        del random_state  # not used (only there for API compatibility)
        # Since we don't have access to column names, we use the member _feature_groups_ids
        X_j = _get_array_cols(X, self._feature_groups_ids[features_group_id])

        scoring_loci = self.scoring(
            self._list_estimators[features_group_id], X_j, y
        )

        return [scoring_loci]

    def _check_fit(self):
        """Check that an estimator has been fitted after removing each group of
        covariates.
        """
        super()._check_fit()
        check_is_fitted(self.estimator_)
        if self._list_estimators is None:
            raise ValueError(
                "The estimators require to be fit before to use them"
            )
        for m in self._list_estimators:
            check_is_fitted(m)


def loci_importance(
    estimator,
    X,
    y,
    scoring=None,
    method=None,
    loss=None,
    feature_groups=None,
    test_statistic="ttest",
    k_best=None,
    percentile=None,
    threshold_min=None,
    threshold_max=None,
    n_jobs: int = 1,
):
    warnings.warn(
        "loci_importance is deprecated and will be removed in version 0.6.0. "
        "Please use class LOCI instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    methods = LOCI(
        estimator=estimator,
        scoring=scoring,
        method=method,
        loss=loss,
        statistical_test=test_statistic,
        feature_groups=feature_groups,
        n_jobs=n_jobs,
    )
    methods.fit_importance(X, y)
    selection = methods.importance_selection(
        k_best=k_best,
        percentile=percentile,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
    )
    return selection, methods.importances_, methods.pvalues_


# use the docstring of the class for the function
loci_importance.__doc__ = _aggregate_docstring(
    [
        LOCI.__doc__,
        LOCI.__init__.__doc__,
        LOCI.fit_importance.__doc__,
        LOCI.importance_selection.__doc__,
    ],
    """
Returns
-------
selection : ndarray of shape (n_groups,)
    Boolean array indicating selected feature groups (True = selected).
importances : ndarray of shape (n_groups,)
    Feature group importance scores/test statistics.
pvalues : ndarray of shape (n_groups,)
    P-values computed for the marginal importance.
""",
)


class LOCICV(BasePerturbationCV):
    """
    Leave-One-Covariate-IN (LOCI) algorithm with Cross-Validation.

    Parameters
    ----------
    estimators: list of sklearn estimators or single sklearn estimator
        Can be a list of fitted sklearn estimators (one per fold) or a single sklearn
        estimator that will then be cloned and fitted on each fold.
    cv: cross-validation generator
        A cross-validation generator object (e.g., KFold, StratifiedKFold).
    statistical_test : callable or str, default="nb-ttest"
        Statistical test function to compute p-values from importance scores.
    scoring : srt, callable
        Strategy to evaluate the performance of the estimator to compute
        importance scores. Based on :func:`sklearn.metrics.check_scoring`.
    method : str, default=None
        The method used for making predictions. This determines the predictions
        passed to the loss function. Supported methods are "predict",
        "predict_proba", "decision_function", "transform".

        .. deprecated:: 0.5.0
            Will be removed in 0.6.0. Please use parameter 'scoring' instead.
    loss : callable, default=None
        The function to compute the loss when comparing the perturbed model
        to the original model.

        .. deprecated:: 0.5.0
            Will be removed in 0.6.0. Please use parameter 'scoring' instead.
    feature_groups: dict or None, default=None
        A dictionary where the keys are the group names and the values are the
        list of column names corresponding to each features group. If None,
        the feature_groups are identified based on the columns of X.
    n_jobs : int, default=1
        The number of jobs to run in parallel. Parallelization is done over the folds.

    Attributes
    ----------
    importance_estimators_ : list of LOCI instances
        The LOCI instances fitted on each fold.
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
        method=None,
        loss=None,
        statistical_test="nb-ttest",
        feature_groups=None,
        n_jobs=1,
    ):
        super().__init__(estimators, cv, statistical_test, n_jobs)
        self.scoring = scoring
        self.method = method
        self.loss = loss
        self.feature_groups = feature_groups

    def _fit_single_split(self, estimator, X_train, y_train):
        """Fit a LOCI instance on a single train/test split."""
        loci = LOCI(
            estimator=estimator,
            scoring=self.scoring,
            feature_groups=self.feature_groups,
            n_jobs=1,  # no parallelization inside the fold
        )
        loci.fit(X_train, y_train)
        return loci
