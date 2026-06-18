import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import check_is_fitted, clone, is_classifier, is_regressor
from sklearn.metrics import mean_squared_error

from hidimstat._utils.docstring import _aggregate_docstring
from hidimstat._utils.utils import check_statistical_test
from hidimstat.base_perturbation import BasePerturbation, BasePerturbationCV


class LOCI(BasePerturbation):
    """
    Leave-One-Covariate-In (LOCO) algorithm

    The model is re-fitted for each single feature/group of features. The importance is
    then computed as the difference between the loss of an empty model (mean for regression,
    and majority vote for classification) and the loss of the model on the single feature/group.
    For more details, see :footcite:t:`ewald_2024`.

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
            clone(self.estimator) for _ in range(self.n_features_groups_)
        ]

        # Parallelize the fitting of the covariate estimators
        self._list_estimators = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_fit_one_features_group)(
                estimator, X, y, key_features_groups
            )
            for key_features_groups, estimator in zip(
                self.features_groups_.keys(),
                self._list_estimators,
                strict=False,
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

        if self.method == "predict":
            y_baseline = np.full_like(y, np.mean(y), dtype=float)
        elif self.method in ["predict_proba", "decision_function"]:
            if y.ndim == 1:
                values, counts = np.unique(y, return_counts=True)
                y_baseline = np.full_like(
                    y, values[np.argmax(counts)], dtype=int
                )
            else:
                # For multilabel classification, we take the marginal probability.
                y_baseline = np.tile(y.mean(axis=0), (len(y), 1))
        else:
            raise ValueError("Estimator must be a classifier or regressor.")
        self.loss_reference_ = self.loss(y, y_baseline)

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
                self.loss_reference_ - self.loss_[j]
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
        """
        Fit the estimator on a group of covariates.
        Used in parallel.
        """
        if isinstance(X, pd.DataFrame):
            X_j = X[self.features_groups_[key_features_group]]
        else:
            X_j = X[:, self.features_groups_[key_features_group]]
        estimator.fit(X_j, y)
        return estimator

    def _joblib_predict_one_features_group(
        self, X, features_group_id, random_state=None
    ):
        """
        Predict the target feature for a single group of covariates.
        Used in parallel.
        """
        del random_state  # not used (only there for API compatibility)
        if isinstance(X, pd.DataFrame):
            X_j = X[self.features_groups_[features_group_id]]
        else:
            # Since we don't have access to column names, we use the member _features_groups_ids
            X_j = X[:, self._features_groups_ids[features_group_id]]

        y_pred_loci = getattr(
            self._list_estimators[features_group_id], self.method
        )(X_j)

        return [y_pred_loci]

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


def loci_importance(
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
    method = LOCI(
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
    selection : ndarray of shape (n_features,)
        Boolean array indicating selected features (True = selected)
    importances : ndarray of shape (n_features,)
        Feature importance scores/test statistics.
    pvalues : ndarray of shape (n_features,)
        None because there is no p-value for this method
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
        """Fit a LOCI instance on a single train/test split."""
        loci = LOCI(
            estimator=estimator,
            method=self.method,
            loss=self.loss,
            features_groups=self.features_groups,
            n_jobs=1,  # no parallelization inside the fold
        )
        loci.fit(X_train, y_train)
        return loci
