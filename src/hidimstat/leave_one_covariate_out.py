import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import check_is_fitted, clone
from sklearn.metrics import root_mean_squared_error

from hidimstat.base_perturbation import BasePerturbation


class LOCO(BasePerturbation):
    def __init__(
        self,
        estimator,
        loss: callable = root_mean_squared_error,
        method: str = "predict",
        feature_groups=None,
        n_jobs: int = 1,
    ):
        """
        Leave-One-Covariate-Out (LOCO) as presented in
        :footcite:t:`lei2018distribution` and :footcite:t:`verdinelli2024feature`.
        The model is re-fitted for each feature/group of features. The importance is
        then computed as the difference between the loss of the full model and the loss
        of the model without the feature/group.

        Parameters
        ----------
        estimator : sklearn compatible estimator
            The estimator to use for the prediction.
        loss : callable, default=root_mean_squared_error
            The loss function to use when comparing the perturbed model to the full
            model.
        method : str, default="predict"
            The method to use for the prediction. This determines the predictions passed
            to the loss function. Supported methods are "predict", "predict_proba" or
            "decision_function".
        feature_groups: dict or None,  default=None
            A dictionary where the keys are the group names and the values are the
            list of column names corresponding to each features group. If None,
            the feature_groups are identified based on the columns of X.
        n_jobs : int, default=1
            The number of jobs to run in parallel. Parallelization is done over the
            features or groups of features.

        Notes
        -----
        :footcite:t:`Williamson_General_2023` also presented a LOCO method with an
        additional data splitting strategy.

        References
        ----------
        .. footbibliography::
        """
        super().__init__(
            estimator=estimator,
            loss=loss,
            method=method,
            n_jobs=n_jobs,
            n_permutations=1,
            feature_groups=feature_groups,
        )
        self._list_estimators = []

    def fit(self, X, y):
        """Fit a model after removing each covariate/group of covariates.

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
                estimator, X, y, key_feature_groups
            )
            for key_feature_groups, estimator in zip(
                self.feature_groups.keys(), self._list_estimators
            )
        )
        return self

    def _joblib_fit_one_features_group(self, estimator, X, y, key_features_group):
        """Fit the estimator after removing a group of covariates. Used in parallel."""
        if isinstance(X, pd.DataFrame):
            X_minus_j = X.drop(columns=self.feature_groups[key_feature_groups])
        else:
            X_minus_j = np.delete(X, self.feature_groups[key_feature_groups], axis=1)
        estimator.fit(X_minus_j, y)
        return estimator

    def _joblib_predict_one_features_group(
        self, X, features_group_id, key_features_group, random_state=None
    ):
        """Predict the target feature after removing a group of covariates.
        Used in parallel."""
        X_minus_j = np.delete(X, self._feature_groups_ids[features_group_id], axis=1)

        y_pred_loco = getattr(self._list_estimators[features_group_id], self.method)(
            X_minus_j
        )

        return [y_pred_loco]

    def _check_fit(self):
        """Check that an estimator has been fitted after removing each group of
        covariates."""
        super()._check_fit()
        check_is_fitted(self.estimator)
        if len(self._list_estimators) == 0:
            raise ValueError("The estimators require to be fit before to use them")
        for m in self._list_estimators:
            check_is_fitted(m)
