import warnings
from functools import partial

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import check_is_fitted, clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

from hidimstat._utils.exception import InternalError
from hidimstat._utils.utils import _check_vim_predict_method
from hidimstat.base_variable_importance import BaseVariableImportance
from hidimstat.statistical_tools.aggregation import quantile_aggregation


class BasePerturbation(BaseVariableImportance):
    def __init__(
        self,
        estimator,
        loss: callable = root_mean_squared_error,
        n_permutations: int = 50,
        method: str = "predict",
        n_jobs: int = 1,
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
        n_jobs : int, default=1
            The number of parallel jobs to run. Parallelization is done over the
            variables or groups of variables.
        """
        super().__init__()
        assert n_permutations > 0, "n_permutations must be positive"
        self.estimator = estimator
        self.loss = loss
        _check_vim_predict_method(method)
        self.method = method
        self.n_jobs = n_jobs
        self.n_permutations = n_permutations
        self.n_groups = None

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
        try:
            check_is_fitted(self.estimator)
        except NotFittedError:
            self.estimator.fit(X, y)
        if groups is None:
            self.n_groups = X.shape[1]
            self.groups = {j: [j] for j in range(self.n_groups)}
            self._groups_ids = np.array(list(self.groups.values()), dtype=int)
        elif isinstance(groups, dict):
            self.n_groups = len(groups)
            self.groups = groups
            if isinstance(X, pd.DataFrame):
                self._groups_ids = []
                for group_key in self.groups.keys():
                    self._groups_ids.append(
                        [
                            i
                            for i, col in enumerate(X.columns)
                            if col in self.groups[group_key]
                        ]
                    )
            else:
                self._groups_ids = [
                    np.array(ids, dtype=int) for ids in list(self.groups.values())
                ]
        else:
            raise ValueError("groups needs to be a dictionary")

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
            for group_id, group_key in enumerate(self.groups.keys())
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
                for j in range(self.n_groups)
            ]
        )
        return out_dict

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
            self.n_groups is None
            or not hasattr(self, "groups")
            or not hasattr(self, "_groups_ids")
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
        for index_variables in self.groups.values():
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
            np.concatenate([values for values in self.groups.values()])
        ).shape[0]
        if X.shape[1] != number_unique_feature_in_groups:
            warnings.warn(
                f"The number of features in X: {X.shape[1]} differs from the"
                " number of features for which importance is computed: "
                f"{number_unique_feature_in_groups}"
            )

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


class BaseCrossValidation:
    """
    Base for Cross validation for feature importance estimation base perturbation.

    This class implements the base of cross validation for feature importance estimation methods.
    It splits the data into folds and fits the feature importance method on each fold,
    then aggregates the results across folds.

    Parameters
    ----------
    cv : cross-validation generator, default=KFold()
        Determines the cross-validation splitting strategy.
    list_parameters : list of dict, optional
        List of parameter dictionaries for each fold. If provided, must match the number
        of folds in cv. Each dictionary contains parameters to initialize feature_importance
        for that specific fold.

    Attributes
    ----------
    list_feature_importance_ : list
        List of fitted feature importance methods for each fold.
    importances_ : ndarray
        Mean feature importance scores across all folds.
    pvalues_ : ndarray
        Aggregated p-values across all folds using quantile aggregation.

    Methods
    -------
    fit(X, y)
        Fit the model using X as training data and y as target values.
    importance(X, y)
        Calculate feature importance scores for X with target y.
    fit_transform(X, y)
        Fit to data, then transform it.
    """

    def __init__(
        self,
        feature_importance,
        estimators,
        cv=KFold(),
        importance_aggregation=partial(np.mean, axis=0),
        pvalue_aggregation=quantile_aggregation,
    ):
        assert issubclass(feature_importance.__class__, BasePerturbation)
        self.feature_importance = feature_importance
        self.cv = cv
        if isinstance(estimators, list):
            if len(estimators) == 0:
                raise ValueError("the list of estimator can be empty")
            elif len(estimators) > 1:
                assert (
                    len(estimators) >= cv.get_n_splits()
                ), "the number of split of the cv should be lower than the number of list of parameters"
            self.estimators = estimators
        else:
            self.estimators = [estimators]

        self.importance_aggregation = importance_aggregation
        self.pvalues_aggregation = pvalue_aggregation

        self.list_feature_importance_ = None
        self.train_index_ = None
        self.test_index_ = None

    def fit(self, X, y):
        """
        Fit the cross-validation model.

        This method performs cross-validation by fitting the feature importance estimator
        on different training folds of the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_sampsrc/hidimstat/base_cross_validation.pyles,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.

        Notes
        -----
        The method creates multiple feature importance estimators, one for each fold
        of the cross-validation, and stores them in the `list_feature_importance_` attribute.
        If list_parameters is provided, each estimator is configured with the corresponding
        parameters for that fold.
        """
        self.list_feature_importance_ = []
        self.train_index_, self.test_index_ = self.cv.split(X)
        for index, train in enumerate(self.train_index_):
            feature_importance = clone(self.feature_importance)
            if len(self.estimators) > 1:
                estimator = self.estimators[index]
            else:
                estimator = clone(self.estimators[0])
            feature_importance.estimator = estimator
            feature_importance.fit(X[train], y[train])
            self.list_feature_importance_.append(feature_importance)
        return self

    def _check_fit(self):
        """
        Checks if feature importances were already calculated.

        This internal method verifies that feature importances have been calculated before allowing
        operations that depend on them. It first checks if importances exist, then calls the parent
        class's importance check.

        Returns
        -------
        bool
            Result of parent class importance check if importances exist

        Raises
        ------
        ValueError
            If feature importances have not been calculated yet (self.list_feature_importance_ is None)
        """
        if (
            self.list_feature_importance_ is None
            or self.train_index_ is None
            or self.test_index_ is None
        ):
            raise ValueError(
                "The importances need to be called before calling this method"
            )

    def importance(self, X, y):
        """Calculate feature importance scores and p-values across cross-validation folds.

        This method computes feature importance scores for each fold in the cross-validation
        and aggregates them to produce final importance scores and p-values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        ndarray of shape (n_features,)
            Averaged feature importance scores across all folds.

        Attributes
        ----------
        importances_ : ndarray of shape (n_features,)
            Mean feature importance scores across all folds.
        pvalues_ : ndarray of shape (n_features,)
            Aggregated p-values across all folds using quantile aggregation.

        Notes
        -----
        This method requires that feature importance calculation has been properly
        initialized through prior method calls.
        """
        self._check_fit()
        for feature_importance, test in zip(
            self.list_feature_importance_,
            self.test_index_,
        ):
            feature_importance.importance(X[test], y[test])
        self.importances_ = self.importance_aggregation(
            [fi.importances_ for fi in self.list_feature_importance_]
        )
        self.pvalues_ = self.pvalues_aggregation(
            np.array([fi.pvalues_ for fi in self.list_feature_importance_])
        )
        return self.importances_

    def fit_importance(self, X, y):
        """
        Fits the model and returns feature importance scores.

        This method combines the functionality of `fit` and `importance` by first fitting
        the model to the data and then calculating feature importance scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Must fulfill input requirements of concrete base model.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        importance_scores : ndarray
            Feature importance scores as calculated by the `importance` method.

        See Also
        --------
        fit : Fits the model to training data
        importance : Calculates feature importance scores

        Notes
        -----
        This is a convenience method that combines model fitting and importance calculation
        in a single call. It is equivalent to calling `fit` followed by `importance`.
        """
        self.fit(X, y)
        return self.importance(X, y)
