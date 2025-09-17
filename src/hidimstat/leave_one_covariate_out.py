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
        estimator : sklearn compatible estimator, optional
            The estimator to use for the prediction.
        loss : callable, default=root_mean_squared_error
            The loss function to use when comparing the perturbed model to the full
            model.
        method : str, default="predict"
            The method to use for the prediction. This determines the predictions passed
            to the loss function. Supported methods are "predict", "predict_proba" or
            "decision_function".
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
        )
        self._list_estimators = []

    def fit(self, X, y, features_groups=None):
        """Fit a model after removing each covariate/group of covariates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        features_groups : dict, default=None
            A dictionary where the keys are the group names and the values are the
            indices of the covariates in each group.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        super().fit(X, y, features_groups)
        # create a list of covariate estimators for each group if not provided
        self._list_estimators = [
            clone(self.estimator) for _ in range(self.n_feature_groups)
        ]

        # Parallelize the fitting of the covariate estimators
        self._list_estimators = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_fit_one_features_group)(
                estimator, X, y, key_features_groups
            )
            for key_features_groups, estimator in zip(
                self.features_groups.keys(), self._list_estimators
            )
        )
        return self

    def _joblib_fit_one_features_group(self, estimator, X, y, key_features_groups):
        """Fit the estimator after removing a group of covariates. Used in parallel."""
        if isinstance(X, pd.DataFrame):
            X_minus_j = X.drop(columns=self.features_groups[key_features_groups])
        else:
            X_minus_j = np.delete(X, self.features_groups[key_features_groups], axis=1)
        estimator.fit(X_minus_j, y)
        return estimator

    def _joblib_predict_one_features_group(
        self, X, feature_group_id, key_features_groups
    ):
        """Predict the target feature after removing a group of covariates.
        Used in parallel."""
        X_minus_j = np.delete(X, self._features_groups_ids[feature_group_id], axis=1)

        y_pred_loco = getattr(self._list_estimators[feature_group_id], self.method)(
            X_minus_j
        )

        return [y_pred_loco]

    def _check_fit(self, X):
        """Check that an estimator has been fitted after removing each group of
        covariates."""
        super()._check_fit(X)
        check_is_fitted(self.estimator)
        if len(self._list_estimators) == 0:
            raise ValueError("The estimators require to be fit before to use them")
        for m in self._list_estimators:
            check_is_fitted(m)
