import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.metrics import root_mean_squared_error

from hidimstat.utils import _check_vim_predict_method


class PermutationImportance(BaseEstimator):
    """
    Parameters
    ----------
    estimator: scikit-learn compatible estimator
        The predictive model.
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
        n_permutations: int = 50,
        loss: callable = root_mean_squared_error,
        method: str = "predict",
        random_state: int = None,
        n_jobs: int = 1,
    ):

        check_is_fitted(estimator)
        self.estimator = estimator
        self.n_permutations = n_permutations

        self.random_state = random_state
        self.loss = loss
        _check_vim_predict_method(method)
        self.method = method
        self.n_jobs = n_jobs
        self.rng = np.random.RandomState(random_state)
        self.n_groups = None

    def fit(self, X, y=None, groups=None):
        """
        Parameters
        ----------
        X: np.ndarray of shape (n_samples, n_features)
            The input samples. Not used here.
        y: np.ndarray of shape (n_samples,)
            The target values. Not used here.
        groups: dict, default=None
            Dictionary of groups for the covariates. The keys are the group names
            and the values are lists of covariate indices.
        """
        self.groups = groups
        return self

    def predict(self, X, y=None):
        """
        Compute the prediction of the model with permuted data for each group.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        y: array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        premuted_y_pred: np.ndarray of shape (n_groups, n_permutations, n_samples)
            The predictions of the model with permuted data for each group

        """
        check_is_fitted(self.estimator)
        if self.groups is None:
            self.n_groups = X.shape[1]
            self.groups = {j: [j] for j in range(self.n_groups)}
        else:
            self.n_groups = len(self.groups)

        def _joblib_predict_one_group(X, j):
            """
            Compute the importance score for a single group of covariates.
            """
            list_y_pred_perm = []
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

            # Create an array X_perm_j of shape (n_permutations, n_samples, n_features)
            # where the j-th group of covariates is permuted
            X_perm_j = np.empty((self.n_permutations, X.shape[0], X.shape[1]))
            X_perm_j[:, :, non_group_ids] = X_minus_j
            # Create the permuted data for the j-th group of covariates
            group_j_permuted = np.array(
                [self.rng.permutation(X_j) for _ in range(self.n_permutations)]
            )
            X_perm_j[:, :, group_ids] = group_j_permuted
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
            delayed(_joblib_predict_one_group)(X, j) for j in self.groups.keys()
        )

        premuted_y_pred = np.stack(out_list, axis=0)
        return premuted_y_pred

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

        output_dict = dict()
        y_pred = getattr(self.estimator, self.method)(X)
        loss_reference = self.loss(y_true=y, y_pred=y_pred)
        output_dict["loss_reference"] = loss_reference
        output_dict["loss_perm"] = dict()

        y_pred_perm = self.predict(X, y)

        output_dict["loss_perm"] = dict()
        for j, y_pred_j in enumerate(y_pred_perm):
            list_loss_perm = []
            for y_pred_perm in y_pred_j:
                list_loss_perm.append(self.loss(y_true=y, y_pred=y_pred_perm))
            output_dict["loss_perm"][j] = np.array(list_loss_perm)

        output_dict["importance"] = np.array(
            [
                np.mean(output_dict["loss_perm"][j]) - output_dict["loss_reference"]
                for j in range(self.n_groups)
            ]
        )

        return output_dict
