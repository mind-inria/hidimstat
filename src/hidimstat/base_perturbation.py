import numpy as np
from joblib import Parallel, delayed
from sklearn.base import check_is_fitted
from sklearn.metrics import root_mean_squared_error

from hidimstat._utils.utils import _check_vim_predict_method, check_random_state
from hidimstat.base_variable_importance import (
    BaseVariableImportance,
    GroupVariableImportanceMixin,
)


class BasePerturbation(BaseVariableImportance, GroupVariableImportanceMixin):
    def __init__(
        self,
        estimator,
        loss: callable = root_mean_squared_error,
        n_permutations: int = 50,
        method: str = "predict",
        feature_groups=None,
        feature_types="auto",
        n_jobs: int = 1,
        random_state=None,
    ):
        """
        Base class for model-agnostic variable importance measures based on
        perturbation.

        Parameters
        ----------
        estimator : sklearn compatible estimator, optional
            The estimator to use for the prediction.
        loss : callable, default=root_mean_squared_error
            The function to compute the loss when comparing the perturbed model
            to the original model.
        n_permutations : int, default=50
            This parameter is relevant only for PFI or CFI.
            Specifies the number of times the variable group (residual for CFI) is
            permuted. For each permutation, the perturbed model's loss is calculated
            and averaged over all permutations.
        method : str, default="predict"
            The method used for making predictions. This determines the predictions
            passed to the loss function. Supported methods are "predict",
            "predict_proba", "decision_function", "transform".
        feature_groups: dict or None, default=None
            A dictionary where the keys are the group names and the values are the
            list of column names corresponding to each features group. If None,
            the feature_groups are identified based on the columns of X.
        feature_types: str or list, default="auto"
            The feature type. Supported types include "auto", "continuous", and
            "categorical". If "auto", the type is inferred from the cardinality
            of the unique values passed to the `fit` method.
        n_jobs : int, default=1
            The number of parallel jobs to run. Parallelization is done over the
            variables or groups of variables.
        random_state : int, default=None
            The random state to use for sampling.
        """
        super().__init__()
        check_is_fitted(estimator)
        assert n_permutations > 0, "n_permutations must be positive"
        self.estimator = estimator
        self.loss = loss
        _check_vim_predict_method(method)
        self.method = method
        self.n_jobs = n_jobs
        self.n_permutations = n_permutations
        GroupVariableImportanceMixin.__init__(
            self, feature_groups=feature_groups, feature_types=feature_types
        )
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Initialize feature groups based on input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like, optional
            The target values. Not used, present for API consistency.
            Defaults to None.

        Returns
        -------
        self : object
            Returns the instance itself to enable method chaining.

        See Also
        --------
        GroupVariableImportanceMixin.fit : Parent class fit method that performs the actual initialization.
        """
        GroupVariableImportanceMixin.fit(self, X, y)
        return self

    def predict(self, X):
        """
        Compute the predictions after perturbation of the data for each group of
        variables.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        out: array-like of shape (n_groups, n_permutations, n_samples)
            The predictions after perturbation of the data for each group of variables.
        """
        self._check_fit(X)
        X_ = np.asarray(X)
        rng = check_random_state(self.random_state)

        # Parallelize the computation of the importance scores for each group
        out_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_predict_one_features_group)(
                X_, feature_group_id, feature_group_key, random_state=child_state
            )
            for feature_group_id, (feature_group_key, child_state) in enumerate(
                zip(self.feature_groups.keys(), rng.spawn(self.n_feature_groups_))
            )
        )
        return np.stack(out_list, axis=0)

    def importance(self, X, y):
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
            - 'loss': a dictionary containing the loss of the perturbed model
            for each group.
            - 'importance': the importance scores for each group.
        """
        self._check_fit(X)

        out_dict = dict()

        y_pred = getattr(self.estimator, self.method)(X)
        loss_reference = self.loss(y, y_pred)
        out_dict["loss_reference"] = loss_reference

        y_pred = self.predict(X)
        out_dict["loss"] = dict()
        for j, y_pred_j in enumerate(y_pred):
            list_loss = []
            for y_pred_perm in y_pred_j:
                list_loss.append(self.loss(y, y_pred_perm))
            out_dict["loss"][j] = np.array(list_loss)

        out_dict["importance"] = np.array(
            [
                np.mean(out_dict["loss"][j]) - loss_reference
                for j in range(self.n_feature_groups_)
            ]
        )
        return out_dict

    def _joblib_predict_one_features_group(
        self, X, feature_group_id, feature_group_key, random_state=None
    ):
        """
        Compute the predictions after perturbation of the data for a given
        group of variables. This function is parallelized.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        feature_group_id: int
            The index of the group of variables.
        feature_group_key: str, int
            The key of the group of variables. (parameter use for debugging)
        random_state:
            The random state to use for sampling.
        """
        feature_group_ids = self._feature_groups_ids[feature_group_id]
        non_feature_group_ids = np.delete(np.arange(X.shape[1]), feature_group_ids)
        # Create an array X_perm_j of shape (n_permutations, n_samples, n_features)
        # where the j-th group of covariates is permuted
        X_perm = np.empty((self.n_permutations, X.shape[0], X.shape[1]))
        X_perm[:, :, non_feature_group_ids] = np.delete(X, feature_group_ids, axis=1)
        X_perm[:, :, feature_group_ids] = self._permutation(
            X, feature_group_id=feature_group_id, random_state=random_state
        )
        # Reshape X_perm to allow for batch prediction
        X_perm_batch = X_perm.reshape(-1, X.shape[1])
        y_pred_perm = getattr(self.estimator, self.method)(X_perm_batch)

        # In case of classification, the output is a 2D array. Reshape accordingly
        if y_pred_perm.ndim == 1:
            y_pred_perm = y_pred_perm.reshape(self.n_permutations, X.shape[0])
        else:
            y_pred_perm = y_pred_perm.reshape(
                self.n_permutations, X.shape[0], y_pred_perm.shape[1]
            )
        return y_pred_perm

    def _permutation(self, X, feature_group_id, random_state=None):
        """Method for creating the permuted data for the j-th group of covariates."""
        raise NotImplementedError
