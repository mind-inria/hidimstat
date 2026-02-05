import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import check_is_fitted, clone
from sklearn.metrics import mean_squared_error

from hidimstat._utils.docstring import _aggregate_docstring
from hidimstat._utils.utils import check_statistical_test
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
    method : str, default="predict"
        The method to use for the prediction. This determines the predictions passed
        to the loss function. Supported methods are "predict", "predict_proba" or
        "decision_function".
    loss : callable, default=mean_squared_error
        The loss function to use when comparing the perturbed model to the full
        model.
    statistical_test : callable or str, default="ttest"
        Statistical test function for computing p-values of importance scores.
    features_groups: dict or None, default=None
        A dictionary where the keys are the group names and the values are the
        list of column names corresponding to each features group. If None,
        the features_groups are identified based on the columns of X.
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
        method: str = "predict",
        loss: callable = mean_squared_error,
        statistical_test="ttest",
        features_groups=None,
        n_jobs: int = 1,
    ):
        super().__init__(
            estimator=estimator,
            method=method,
            loss=loss,
            n_permutations=1,
            statistical_test=statistical_test,
            features_groups=features_groups,
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
            clone(self.estimator) for _ in range(self.n_features_groups_)
        ]

        # Parallelize the fitting of the covariate estimators
        self._list_estimators = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_fit_one_features_group)(
                estimator, X, y, key_features_groups
            )
            for key_features_groups, estimator in zip(
                self.features_groups_.keys(), self._list_estimators
            )
        )
        return self

    def importance(self, X, y):
        """
        Compute the importance scores for each group of covariates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to compute importance scores for.
        y : array-like of shape (n_samples,)

        importances_ : ndarray of shape (n_groups,)
            The importance scores for each group of covariates.
            A higher score indicates greater importance of that group.

        Returns
        -------
        importances_ : ndarray of shape (n_features,)
            Importance scores for each feature.

        Attributes
        ----------
        loss_reference_ : float
            The loss of the model with the original (non-perturbed) data.
        loss_ : dict
            Dictionary with indices as keys and arrays of perturbed losses as values.
            Contains the loss values for each permutation of each group.
        importances_ : ndarray of shape (n_groups,)
            The calculated importance scores for each group.
        pvalues_ : ndarray of shape (n_groups,)
            P-values from one-sided t-test testing if importance scores are
            significantly greater than 0.

        Notes
        -----
        The importance score for each group is calculated as the mean increase in loss
        when that group is perturbed, compared to the reference loss.
        A higher importance score indicates that perturbing that group leads to
        worse model performance, suggesting those features are more important.
        """
        self._check_fit()
        self._check_compatibility(X)
        statistical_test = check_statistical_test(self.statistical_test)

        y_pred = getattr(self.estimator, self.method)(X)
        self.loss_reference_ = self.loss(y, y_pred)

        y_pred = self._predict(X)
        test_result = []
        self.loss_ = {}
        for j, y_pred_j in enumerate(y_pred):
            self.loss_[j] = np.array([self.loss(y, y_pred_j[0])])
            if np.all(np.equal(y.shape, y_pred_j[0].shape)):
                test_result.append(y - y_pred_j[0])
            else:
                test_result.append(
                    y - np.unique(y)[np.argmax(y_pred_j[0], axis=-1)]
                )

        self.importances_ = np.mean(
            [
                self.loss_[j] - self.loss_reference_
                for j in range(self.n_features_groups_)
            ],
            axis=1,
        )
        self.pvalues_ = statistical_test(np.array(test_result)).pvalue
        assert self.pvalues_.shape[0] == y_pred.shape[0], (
            "The statistical test doesn't provide the correct dimension."
        )
        return self.importances_

    def _joblib_fit_one_features_group(
        self, estimator, X, y, key_features_group
    ):
        """Fit the estimator after removing a group of covariates. Used in parallel."""
        if isinstance(X, pd.DataFrame):
            X_minus_j = X.drop(
                columns=self.features_groups_[key_features_group]
            )
        else:
            X_minus_j = np.delete(
                X, self.features_groups_[key_features_group], axis=1
            )
        estimator.fit(X_minus_j, y)
        return estimator

    def _joblib_predict_one_features_group(
        self, X, features_group_id, random_state=None
    ):
        """Predict the target feature after removing a group of covariates.
        Used in parallel.
        """
        X_minus_j = np.delete(
            X, self._features_groups_ids[features_group_id], axis=1
        )

        y_pred_loco = getattr(
            self._list_estimators[features_group_id], self.method
        )(X_minus_j)

        return [y_pred_loco]

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
    method: str = "predict",
    loss: callable = mean_squared_error,
    features_groups=None,
    test_statistic="ttest",
    k_best=None,
    percentile=None,
    threshold_min=None,
    threshold_max=None,
    n_jobs: int = 1,
):
    method = LOCO(
        estimator=estimator,
        method=method,
        loss=loss,
        statistical_test=test_statistic,
        features_groups=features_groups,
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
    selection : ndarray of shape (n_features,)
        Boolean array indicating selected features (True = selected)
    importances : ndarray of shape (n_features,)
        Feature importance scores/test statistics.
    pvalues : ndarray of shape (n_features,)
        None because there is no p-value for this method
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
    method : str, default="predict"
        The method to use for the prediction. This determines the predictions passed
        to the loss function. Supported methods are "predict", "predict_proba" or
        "decision_function".
    loss : callable, default=mean_squared_error
        The loss function to use when comparing the perturbed model to the full
        model.
    features_groups: dict or None, default=None
        A dictionary where the keys are the group names and the values are the
        list of column names corresponding to each features group. If None,
        the features_groups are identified based on the columns of X.
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
        statistical_test="nb-ttest",
        method="predict",
        loss=mean_squared_error,
        features_groups=None,
        n_jobs=1,
    ):
        super().__init__(estimators, cv, statistical_test, n_jobs)
        self.method = method
        self.loss = loss
        self.features_groups = features_groups

    def _fit_single_split(self, estimator, X_train, y_train):
        """Fit a LOCO instance on a single train/test split."""
        loco = LOCO(
            estimator=estimator,
            method=self.method,
            loss=self.loss,
            features_groups=self.features_groups,
            n_jobs=1,  # no parallelization inside the fold
        )
        loco.fit(X_train, y_train)
        return loco
