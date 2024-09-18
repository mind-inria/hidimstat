"""
Re-implementation of the block-based CPI algorithm separating the predictive
model fit / predict from the model inspection.

Chamma, A., Engemann, D., & Thirion, B. (2023). Statistically Valid Variable
Importance Assessment through Conditional Permutations. In Proceedings of the
37th Conference on Neural Information Processing Systems (NeurIPS 2023)
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.base import check_is_fitted
from sklearn.base import clone
from sklearn.metrics import mean_squared_error


class CPI(BaseEstimator, TransformerMixin):
    """
    Conditional Permutation Importance (CPI) algorithm.

    Parameters
    ----------
    estimator: scikit-learn compatible estimator
        The predictive model.
    covariate_estimator: scikit-learn compatible estimator or list of
    estimators
        The model(s) used to estimate the covariates. If a single estimator is
        provided, it will be cloned for each covariate. Otherwise, a list of
        potentially different estimators can be provided, the length of the
        list must match the number of covariates.
    n_perm: int, default=50
        Number of permutations to perform.
    groups: dict, default=None
        Dictionary of groups for the covariates. The keys are the group names
        and the values are lists of covariate indices.
        It is assumed that the covariates are contiguous within each group.
    loss: callable, default=mean_squared_error
        Loss function to evaluate the model performance.
    score_proba: bool, default=False
        Whether to use the predict_proba method of the estimator.
    random_state: int, default=None
        Random seed for the permutation.
    scoring: callable, default=None
        Scoring function to evaluate the model performance.
    """

    def __init__(self,
                 estimator,
                 covariate_estimator,
                 n_perm: int = 50,
                 groups: dict = None,
                 loss: callable = mean_squared_error,
                 score_proba: bool = False,
                 random_state: int = None,
                 ):

        check_is_fitted(estimator)
        self.estimator = estimator
        self.covariate_estimator = covariate_estimator
        if isinstance(self.covariate_estimator, list):
            self.list_cov_estimators = self.covariate_estimator
        else:
            self.list_cov_estimators = []
        self.n_perm = n_perm
        self.groups = groups
        self.random_state = random_state
        self.loss = loss
        self.score_proba = score_proba

        self.rng = np.random.RandomState(random_state)

    def fit(self, X, y):
        """
        Fit the covariate estimators to predict each group of covariates from
        the others.
        """
        if self.groups is None:
            self.nb_groups = X.shape[1]
            self.groups = {j: [j] for j in range(self.nb_groups)}
        else:
            self.nb_groups = len(self.groups)
        # create a list of covariate estimators for each group if not provided
        if len(self.list_cov_estimators) == 0:
            self.list_cov_estimators = [clone(self.covariate_estimator)
                                        for _ in range(self.nb_groups)]
        # fit the covariate estimators on each group
        for j in range(self.nb_groups):
            X_j = X[:, self.groups[j]].copy()
            X_minus_j = np.delete(X, self.groups[j], axis=1)
            self.list_cov_estimators[j].fit(X_minus_j, X_j)

        return self

    def predict(self, X, y):
        """
        Compute the CPI importance scores. For each group of covariates, the
        residuals are computed using the covariate estimators. The residuals
        are then permuted and the model is re-evaluated. The importance score
        is the difference between the loss of the model with the original data
        and the loss of the model with the permuted data.
        """
        if len(self.list_cov_estimators) == 0:
            raise ValueError("fit must be called before predict")
        for m in self.list_cov_estimators:
            check_is_fitted(m)

        output_dict = dict()
        if self.score_proba:
            y_pred = self.estimator.predict_proba(X)
        else:
            y_pred = self.estimator.predict(X)
        loss_reference = self.loss(y_true=y, y_pred=y_pred)
        output_dict["loss_reference"] = loss_reference
        output_dict['loss_perm'] = dict()

        for j in range(self.nb_groups):
            list_loss_perm = []
            X_j = X[:, self.groups[j]].copy()
            X_minus_j = np.delete(X, self.groups[j], axis=1)
            X_j_hat = self.list_cov_estimators[j].predict(
                X_minus_j).reshape(X_j.shape)
            residual_j = X_j - X_j_hat
            for _ in range(self.n_perm):
                X_j_perm = X_j_hat + self.rng.permutation(residual_j)
                # Currently assumes that each group is contiguous
                start_group_idx = np.min(self.groups[j])

                X_perm = np.insert(
                    X_minus_j,
                    [start_group_idx] * len(self.groups[j]),
                    X_j_perm,
                    axis=1
                )

                if self.score_proba:
                    y_pred_perm = self.estimator.predict_proba(X_perm)
                else:
                    y_pred_perm = self.estimator.predict(X_perm)
                list_loss_perm.append(self.loss(y_true=y, y_pred=y_pred_perm))
            output_dict['loss_perm'][j] = list_loss_perm

        output_dict['importance'] = np.array([
            np.mean(output_dict['loss_perm'][j] -
                    output_dict['loss_reference'])
            for j in range(self.nb_groups)])

        return output_dict
