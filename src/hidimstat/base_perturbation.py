import numpy as np
from joblib import Parallel, delayed
from sklearn.base import check_is_fitted
from sklearn.metrics import mean_squared_error

from hidimstat._utils.utils import (
    _check_vim_predict_method,
    check_random_state,
    check_statistical_test,
)
from hidimstat.base_variable_importance import (
    BaseVariableImportance,
    GroupVariableImportanceMixin,
)


class BasePerturbation(BaseVariableImportance, GroupVariableImportanceMixin):
    """
    Abstract base class for model-agnostic variable importance measures using
    perturbation techniques.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        The fitted estimator used for predictions.
    method : str, default="predict"
        The method used for making predictions. This determines the predictions
        passed to the loss function. Supported methods are "predict",
        "predict_proba", "decision_function", "transform".
    loss : callable, default=mean_squared_error
        The function to compute the loss when comparing the perturbed model
        to the original model.
    n_permutations : int, default=50
        Number of permutations for each feature group.
    statistical_test : callable or str, default="wilcoxon"
        Statistical test function for computing p-values of importance scores.
    features_groups : dict or None, default=None
        Mapping of group names to lists of feature indices or names. If None, groups are inferred.
    n_jobs : int, default=1
        Number of parallel jobs for computation.
    random_state : int or None, default=None
        Seed for reproducible permutations.

    Attributes
    ----------
    features_groups : dict
        Mapping of feature groups identified during fit.
    importances_ : ndarray (n_groups,)
        Importance scores for each feature group.
    loss_reference_ : float
        Loss on original (non-perturbed) data.
    loss_ : dict
        Loss values for each permutation of each group.
    pvalues_ : ndarray of shape (n_groups,)
        P-values for importance scores.

    Notes
    -----
    This class is abstract. Subclasses must implement the `_permutation` method
    to define how feature groups are perturbed.
    """

    def __init__(
        self,
        estimator,
        method: str = "predict",
        loss: callable = mean_squared_error,
        n_permutations: int = 50,
        statistical_test="wilcoxon",
        features_groups=None,
        n_jobs: int = 1,
        random_state=None,
    ):
        super().__init__()
        GroupVariableImportanceMixin.__init__(self, features_groups=features_groups)
        check_is_fitted(estimator)
        assert n_permutations > 0, "n_permutations must be positive"
        self.estimator = estimator
        self.loss = loss
        _check_vim_predict_method(method)
        self.method = method
        self.n_permutations = n_permutations
        self.statistical_test = check_statistical_test(statistical_test)
        self.n_jobs = n_jobs

        # variable set in importance
        self.loss_reference_ = None
        self.loss_ = None
        # internal variables
        self._n_groups = None
        self._groups_ids = None
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
        hidimstat.base_variable_importance.GroupVariableImportanceMixin.fit : Parent class fit method that performs the actual initialization.
        """
        GroupVariableImportanceMixin.fit(self, X, y)
        return self

    def _check_fit(self):
        """Check if the instance has been fitted."""
        GroupVariableImportanceMixin._check_fit(self)

    def _check_compatibility(self, X):
        """Check compatibility between input data and fitted model."""
        GroupVariableImportanceMixin._check_compatibility(self, X)

    def _predict(self, X):
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
        X_ = np.asarray(X)
        rng = check_random_state(self.random_state)

        # Parallelize the computation of the importance scores for each group
        out_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_predict_one_features_group)(
                X_, features_group_id, features_group_key, random_state=child_state
            )
            for features_group_id, (features_group_key, child_state) in enumerate(
                zip(self.features_groups.keys(), rng.spawn(self.n_features_groups_))
            )
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
            P-values from one-sample t-test testing if importance scores are
            significantly greater than 0.

        Notes
        -----
        The importance score for each group is calculated as the mean increase in loss
        when that group is perturbed, compared to the reference loss.
        A higher importance score indicates that perturbing that group leads to
        worse model performance, suggesting those features are more important.
        """
        GroupVariableImportanceMixin._check_fit(self)
        GroupVariableImportanceMixin._check_compatibility(self, X)
        self._check_fit()

        y_pred = getattr(self.estimator, self.method)(X)
        self.loss_reference_ = self.loss(y, y_pred)

        y_pred = self._predict(X)
        self.loss_ = dict()
        for j, y_pred_j in enumerate(y_pred):
            list_loss = []
            for y_pred_perm in y_pred_j:
                list_loss.append(self.loss(y, y_pred_perm))
            self.loss_[j] = np.array(list_loss)

        test_result = np.array(
            [
                self.loss_[j] - self.loss_reference_
                for j in range(self.n_features_groups_)
            ]
        )
        self.importances_ = np.mean(test_result, axis=1)
        self.pvalues_ = self.statistical_test(test_result).pvalue
        return self.importances_

    def fit_importance(self, X, y):
        """
        Fits the model to the data and computes feature importance scores.
        Convenience method that combines fit() and importance() into a single call.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

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
        self.fit(X, y)
        return self.importance(X, y)

    def _check_importance(self):
        """
        Checks if the loss has been computed.
        """
        super()._check_importance()
        if (self.loss_reference_ is None) or (self.loss_ is None):
            raise ValueError("The importance method has not yet been called.")

    def _joblib_predict_one_features_group(
        self, X, features_group_id, features_group_key, random_state=None
    ):
        """
        Compute the predictions after perturbation of the data for a given
        group of variables. This function is parallelized.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        features_group_id: int
            The index of the group of variables.
        features_group_key: str, int
            The key of the group of variables. (parameter use for debugging)
        random_state:
            The random state to use for sampling.
        """
        features_group_ids = self._features_groups_ids[features_group_id]
        non_features_group_ids = np.delete(np.arange(X.shape[1]), features_group_ids)
        # Create an array X_perm_j of shape (n_permutations, n_samples, n_features)
        # where the j-th group of covariates is permuted
        X_perm = np.empty((self.n_permutations, X.shape[0], X.shape[1]))
        X_perm[:, :, non_features_group_ids] = np.delete(X, features_group_ids, axis=1)
        X_perm[:, :, features_group_ids] = self._permutation(
            X, features_group_id=features_group_id, random_state=random_state
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

    def _permutation(self, X, features_group_id, random_state=None):
        """Method for creating the permuted data for the j-th group of covariates."""
        raise NotImplementedError
