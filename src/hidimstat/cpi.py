import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, check_is_fitted, clone
from sklearn.metrics import root_mean_squared_error

from hidimstat.utils import _check_vim_predict_method


class CPI(BaseEstimator):
    """
    Conditional Permutation Importance (CPI) algorithm.
    :footcite:t:`Chamma_NeurIPS2023` and for group-level see
    :footcite:t:`Chamma_AAAI2024`.

    Parameters
    ----------
    estimator: scikit-learn compatible estimator
        The predictive model.
    imputation_model: scikit-learn compatible estimator or list of estimators
        The model(s) used to estimate the covariates. If a single estimator is
        provided, it will be cloned for each covariate group. Otherwise, a list of
        potentially different estimators can be provided, the length of the
        list must match the number of covariate groups.
    n_permutations: int, default=50
        Number of permutations to perform.
    loss: callable, default=root_mean_squared_error
        Loss function to evaluate the model performance.
    method: str, default='predict'
        Method to use for predicting values that will be used to compute
        the loss and the importance scores. The method must be implemented by the
        estimator. Supported methods are 'predict', 'predict_proba',
        'decision_function' and 'transform'.
    random_state: int, default=None
        Random seed for the permutation.
    n_jobs: int, default=1
        Number of jobs to run in parallel.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        estimator,
        imputation_model,
        n_permutations: int = 50,
        loss: callable = root_mean_squared_error,
        method: str = "predict",
        random_state: int = None,
        n_jobs: int = 1,
    ):

        check_is_fitted(estimator)
        self.estimator = estimator
        self.imputation_model = imputation_model

        self.n_permutations = n_permutations
        self.random_state = random_state
        self.loss = loss
        _check_vim_predict_method(method)
        self.method = method
        self.n_jobs = n_jobs
        self._list_imputation_models = []

        self.rng = np.random.RandomState(random_state)

    def fit(self, X, y=None, groups=None):
        """
        Fit the covariate estimators to predict each group of covariates from
        the others.

        Parameters
        ----------
        X: np.ndarray of shape (n_samples, n_features)
            The input samples. If groups is provided, the columns must correspond to
            the values of the groups dictionary.
        y: np.ndarray of shape (n_samples,)
            The target values. Not used in the fitting of the covariate estimators.
        groups: dict, default=None
            Dictionary of groups for the covariates. The keys are the group names
            and the values are lists of covariate indices.
        """
        self.groups = groups
        if isinstance(self.imputation_model, list):
            self._list_imputation_models = self.imputation_model
        else:
            self._list_imputation_models = []

        if self.groups is None:
            self.n_groups = X.shape[1]
            self.groups = {j: [j] for j in range(self.n_groups)}
        else:
            self.n_groups = len(self.groups)
        # create a list of covariate estimators for each group if not provided
        if len(self._list_imputation_models) == 0:
            self._list_imputation_models = [
                clone(self.imputation_model) for _ in range(self.n_groups)
            ]

        def _joblib_fit_one_group(estimator, X, y, j):
            """
            Fit a single covariate estimator to predict a single group of
            covariates.
            """
            if isinstance(X, pd.DataFrame):
                X_j = X[self.groups[j]].copy().values
                X_minus_j = X.drop(columns=self.groups[j]).values
            else:
                X_j = X[:, self.groups[j]].copy()
                X_minus_j = np.delete(X, self.groups[j], axis=1)
            estimator.fit(X_minus_j, X_j)
            return estimator

        # Parallelize the fitting of the covariate estimators
        self._list_imputation_models = Parallel(n_jobs=self.n_jobs)(
            delayed(_joblib_fit_one_group)(estimator, X, y, j)
            for j, estimator in zip(self.groups.keys(), self._list_imputation_models)
        )

        return self

    def predict(self, X, y=None):
        """
        Compute the prediction of the model with perturbed data for each group.
        For each group of covariates, the residuals are computed using the
        covariate estimators. The residuals are then permuted and the model is
        re-evaluated.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        y: array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        residual_permuted_y_pred: np.ndarray of shape (n_groups, n_permutations, n_samples)
            The predictions of the model with conditional permutation for each group
        """
        if len(self._list_imputation_models) == 0:
            raise ValueError("fit must be called before predict")
        for m in self._list_imputation_models:
            check_is_fitted(m)

        def _joblib_predict_one_group(imputation_model, X, j):
            """
            Compute the prediction of the model with the permuted data for a
            single group of covariates.
            """

            if isinstance(X, pd.DataFrame):
                X_j = X[self.groups[j]].copy().values
                X_minus_j = X.drop(columns=self.groups[j]).values
                group_ids = [
                    i for i, col in enumerate(X.columns) if col in self.groups[j]
                ]
                non_group_ids = [
                    i for i, col in enumerate(X.columns) if col not in self.groups[j]
                ]
            else:
                X_j = X[:, self.groups[j]].copy()
                X_minus_j = np.delete(X, self.groups[j], axis=1)
                group_ids = self.groups[j]
                non_group_ids = np.delete(np.arange(X.shape[1]), group_ids)

            X_j_hat = imputation_model.predict(X_minus_j).reshape(X_j.shape)
            residual_j = X_j - X_j_hat

            # Create an array X_perm_j of shape (n_permutations, n_samples, n_features)
            # where the j-th group of covariates is (conditionally) permuted
            X_perm_j = np.empty((self.n_permutations, X.shape[0], X.shape[1]))
            X_perm_j[:, :, non_group_ids] = X_minus_j
            # Create the permuted data for the j-th group of covariates
            residual_j_perm = np.array(
                [self.rng.permutation(residual_j) for _ in range(self.n_permutations)]
            )
            X_perm_j[:, :, group_ids] = X_j_hat[np.newaxis, :, :] + residual_j_perm
            # Reshape X_perm_j to allow for batch prediction
            X_perm_batch = X_perm_j.reshape(-1, X.shape[1])
            if isinstance(X, pd.DataFrame):
                X_perm_batch = pd.DataFrame(
                    X_perm_batch.reshape(-1, X.shape[1]), columns=X.columns
                )
            y_pred_perm = getattr(self.estimator, self.method)(X_perm_batch)

            # In case of classification, the output is a 2D array. Reshape accordingly
            if y_pred_perm.ndim == 1:
                y_pred_perm = y_pred_perm.reshape(self.n_permutations, X.shape[0])
            else:
                y_pred_perm = y_pred_perm.reshape(
                    self.n_permutations, X.shape[0], y_pred_perm.shape[1]
                )
            return y_pred_perm

        # Parallelize the computation of the importance scores for each group
        out_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_joblib_predict_one_group)(imputation_model, X, j)
            for j, imputation_model in zip(
                self.groups.keys(), self._list_imputation_models
            )
        )

        residual_permuted_y_pred = np.stack(out_list, axis=0)
        return residual_permuted_y_pred

    def score(self, X, y):
        """
        Compute the importance scores for each group of covariates.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        y: array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        out_dict: dict
            A dictionary containing the following keys:
            - 'loss_reference': the loss of the model with the original data.
            - 'loss_perm': a dictionary containing the loss of the model with
            the permuted data for each group.
            - 'importance': the importance scores for each group.
        """
        check_is_fitted(self.estimator)
        if len(self._list_imputation_models) == 0:
            raise ValueError("fit must be called before score")
        for m in self._list_imputation_models:
            check_is_fitted(m)

        out_dict = dict()

        y_pred = getattr(self.estimator, self.method)(X)

        # In sklearn API y_true is the first argument. Not specifying `y_true=...`
        # will allows using other losses such as `hinge_loss`.
        loss_reference = self.loss(y, y_pred)
        out_dict["loss_reference"] = loss_reference

        y_pred_perm = self.predict(X, y)

        out_dict["loss_perm"] = dict()
        for j, y_pred_j in enumerate(y_pred_perm):
            list_loss_perm = []
            for y_pred_perm in y_pred_j:
                list_loss_perm.append(self.loss(y, y_pred_perm))
            out_dict["loss_perm"][j] = np.array(list_loss_perm)

        out_dict["importance"] = np.array(
            [
                np.mean(out_dict["loss_perm"][j]) - loss_reference
                for j in range(self.n_groups)
            ]
        )

        return out_dict
