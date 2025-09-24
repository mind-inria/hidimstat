from functools import partial
import warnings


import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.base import check_is_fitted
from sklearn.metrics import root_mean_squared_error

from hidimstat._utils.utils import _check_vim_predict_method
from hidimstat._utils.exception import InternalError
from hidimstat.base_variable_importance import BaseVariableImportance


class BasePerturbation(BaseVariableImportance):
    """
    Base class for model-agnostic variable importance measures based on
    perturbation.

    Parameters
    ----------
    estimator : sklearn compatible estimator
        The estimator to use for the prediction.
    method : str, default="predict"
        The method used for making predictions. This determines the predictions
        passed to the loss function. Supported methods are "predict",
        "predict_proba", "decision_function", "transform".
    loss : callable, default=root_mean_squared_error
        Loss function to compute difference between original and perturbed predictions.
    n_permutations : int, default=50
        Number of permutations to perform for calculating variable importance.
        Higher values give more stable results but increase computation time.
    n_jobs : int, default=1
        Number of parallel jobs to run. -1 means using all processors.

    Attributes
    ----------
    features_groups : dict
        Mapping of feature groups identified during fit.
    importances_ : ndarray
        Computed importance scores for each feature group.
    loss_reference_ : float
        Loss of the original model without perturbation.
    loss_ : dict
        Loss values for each perturbed feature group.
    pvalues_ : ndarray
        P-values for importance scores.

    Notes
    -----
    This is an abstract base class. Concrete implementations must override
    the _permutation method.
    """

    def __init__(
        self,
        estimator,
        method: str = "predict",
        loss: callable = root_mean_squared_error,
        n_permutations: int = 50,
        test_statict=partial(wilcoxon, axis=1),
        n_jobs: int = 1,
    ):

        super().__init__()
        check_is_fitted(estimator)
        assert n_permutations > 0, "n_permutations must be positive"
        self.estimator = estimator
        self.loss = loss
        _check_vim_predict_method(method)
        self.method = method
        self.n_permutations = n_permutations
        self.test_statistic = test_statict
        self.n_jobs = n_jobs
        # variable set in fit
        self.features_groups = None
        # varaible set in importance
        self.loss_reference_ = None
        self.loss_ = None
        # internal variables
        self._n_groups = None
        self._groups_ids = None

    def fit(self, X, y=None, groups=None):
        """Base fit method for perturbation-based methods. Identifies the groups.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        y: array-like of shape (n_samples,)
            Not used, only present for consistency with the sklearn API.
        groups: dict, optional
            A dictionary where the keys are the group names and the values are the
            list of column names corresponding to each group. If None, the groups are
            identified based on the columns of X.
        """
        if groups is None:
            self._n_groups = X.shape[1]
            self.features_groups = {j: [j] for j in range(self._n_groups)}
            self._groups_ids = np.array(list(self.features_groups.values()), dtype=int)
        elif isinstance(groups, dict):
            self._n_groups = len(groups)
            self.features_groups = groups
            if isinstance(X, pd.DataFrame):
                self._groups_ids = []
                for group_key in self.features_groups.keys():
                    self._groups_ids.append(
                        [
                            i
                            for i, col in enumerate(X.columns)
                            if col in self.features_groups[group_key]
                        ]
                    )
            else:
                self._groups_ids = [
                    np.array(ids, dtype=int)
                    for ids in list(self.features_groups.values())
                ]
        else:
            raise ValueError("groups needs to be a dictionnary")
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

        # Parallelize the computation of the importance scores for each group
        out_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_predict_one_group)(X_, group_id, group_key)
            for group_id, group_key in enumerate(self.features_groups.keys())
        )
        return np.stack(out_list, axis=0)

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
        self._check_fit(X)

        y_pred = getattr(self.estimator, self.method)(X)
        self.loss_reference_ = self.loss(y, y_pred)

        y_pred = self.predict(X)
        self.loss_ = dict()
        for j, y_pred_j in enumerate(y_pred):
            list_loss = []
            for y_pred_perm in y_pred_j:
                list_loss.append(self.loss(y, y_pred_perm))
            self.loss_[j] = np.array(list_loss)

        test_result = np.array(
            [self.loss_[j] - self.loss_reference_ for j in range(self._n_groups)]
        )
        self.importances_ = np.mean(test_result, axis=1)
        self.pvalues_ = self.test_statistic(test_result).pvalue
        return self.importances_

    def fit_importance(self, X, y, groups=None):
        """
        Fits the model to the data and computes feature importance scores.
        Convenience method that combines fit() and importance() into a single call.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        groups: dict, optional
            A dictionary where the keys are the group names and the values are the
            list of column names corresponding to each group. If None, the groups are
            identified based on the columns of X.

        Returns
        -------
        importances_ : ndarray of shape (n_groups,)
            The calculated importance scores for each feature group.
            Higher values indicate greater importance.

        Notes
        -----
        This method first calls fit() to identify feature groups, then calls
        importance() to compute the importance scores for each group.
        """
        self.fit(X, y, groups)
        return self.importance(X, y)

    def _check_fit(self, X):
        """
        Check if the perturbation method has been properly fitted.

        This method verifies that the perturbation method has been fitted by checking
        if required attributes are set and if the number of features matches
        the grouped variables.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to validate against the fitted model.

        Raises
        ------
        ValueError
            If the method has not been fitted (i.e., if n_groups, groups,
            or _groups_ids attributes are missing).
        AssertionError
            If the number of features in X does not match the total number
            of features in the grouped variables.
        """
        if (
            self._n_groups is None
            or self.features_groups is None
            or self._groups_ids is None
        ):
            raise ValueError(
                "The class is not fitted. The fit method must be called"
                " to set variable groups. If no grouping is needed,"
                " call fit with groups=None"
            )
        if isinstance(X, pd.DataFrame):
            names = list(X.columns)
        elif isinstance(X, np.ndarray) and X.dtype.names is not None:
            names = X.dtype.names
            # transform Structured Array in pandas array for a better manipulation
            X = pd.DataFrame(X)
        elif isinstance(X, np.ndarray):
            names = None
        else:
            raise ValueError("X should be a pandas dataframe or a numpy array.")
        number_columns = X.shape[1]
        for index_variables in self.features_groups.values():
            if type(index_variables[0]) is int or np.issubdtype(
                type(index_variables[0]), int
            ):
                assert np.all(
                    np.array(index_variables, dtype=int) < number_columns
                ), "X does not correspond to the fitting data."
            elif type(index_variables[0]) is str or np.issubdtype(
                type(index_variables[0]), str
            ):
                assert np.all(
                    [name in names for name in index_variables]
                ), f"The array is missing at least one of the following columns {index_variables}."
            else:
                raise InternalError(
                    "A problem with indexing has happened during the fit."
                )
        number_unique_feature_in_groups = np.unique(
            np.concatenate([values for values in self.features_groups.values()])
        ).shape[0]
        if X.shape[1] != number_unique_feature_in_groups:
            warnings.warn(
                f"The number of features in X: {X.shape[1]} differs from the"
                " number of features for which importance is computed: "
                f"{number_unique_feature_in_groups}"
            )

    def _check_importance(self):
        """
        Checks if the loss has been computed.
        """
        super()._check_importance()
        if (self.loss_reference_ is None) or (self.loss_ is None):
            raise ValueError("The importance method has not yet been called.")

    def _joblib_predict_one_group(self, X, group_id, group_key):
        """
        Compute the predictions after perturbation of the data for a given
        group of variables. This function is parallelized.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        group_id: int
            The index of the group of variables.
        group_key: str, int
            The key of the group of variables. (parameter use for debugging)
        """
        group_ids = self._groups_ids[group_id]
        non_group_ids = np.delete(np.arange(X.shape[1]), group_ids)
        # Create an array X_perm_j of shape (n_permutations, n_samples, n_features)
        # where the j-th group of covariates is permuted
        X_perm = np.empty((self.n_permutations, X.shape[0], X.shape[1]))
        X_perm[:, :, non_group_ids] = np.delete(X, group_ids, axis=1)
        X_perm[:, :, group_ids] = self._permutation(X, group_id=group_id)
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

    def _permutation(self, X, group_id):
        """Method for creating the permuted data for the j-th group of covariates."""
        raise NotImplementedError
