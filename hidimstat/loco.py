import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted, clone
from sklearn.metrics import root_mean_squared_error


class LOCO(BaseEstimator, TransformerMixin):
    """
    Leave-One-Covariate-Out (LOCO) algorithm as described in
    :footcite:t:`Chamma_NeurIPS2023`.


    Parameters
    ----------
    estimator: scikit-learn compatible estimator
        The predictive model.
    loss: callable, default=root_mean_squared_error
        Loss function to evaluate the model performance.
    score_proba: bool, default=False
        Whether to use the predict_proba method of the estimator.
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
        loss: callable = root_mean_squared_error,
        score_proba: bool = False,
        random_state: int = None,
        n_jobs: int = 1,
    ):

        check_is_fitted(estimator)
        self.estimator = estimator
        self.random_state = random_state
        self.loss = loss
        self.score_proba = score_proba
        self.n_jobs = n_jobs
        self.rng = np.random.RandomState(random_state)
        self._list_estimators = []

    def fit(self, X, y, groups=None):
        """
        Fit the estimators on each subset of covariates.

        Parameters
        ----------
        groups: dict, default=None
            Dictionary of groups for the covariates. The keys are the group names
            and the values are lists of covariate indices.
        """
        self.groups = groups
        if self.groups is None:
            self.n_groups = X.shape[1]
            self.groups = {j: [j] for j in range(self.n_groups)}
        else:
            self.n_groups = len(self.groups)
        # create a list of covariate estimators for each group if not provided

        self._list_estimators = [clone(self.estimator) for _ in range(self.n_groups)]

        def _joblib_fit_one_group(estimator, X, y, j):
            """
            Fit a single model on a subset of covariates.
            """
            X_minus_j = np.delete(X, self.groups[j], axis=1)
            estimator.fit(X_minus_j, y)
            return estimator

        # Parallelize the fitting of the covariate estimators
        self._list_estimators = Parallel(n_jobs=self.n_jobs)(
            delayed(_joblib_fit_one_group)(estimator, X, y, j)
            for j, estimator in enumerate(self._list_estimators)
        )

        return self

    def predict(self, X, y):
        """

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        y: array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        output_dict: dict
            A dictionary containing the following keys:
            - 'loss_reference': the loss of the model with the original data.
            - 'loss_loco': a dictionary containing the loss of the model with
                each covariate group removed.
            - 'importance': the importance scores for each group.
        """
        check_is_fitted(self.estimator)
        for m in self._list_estimators:
            check_is_fitted(m)

        output_dict = dict()
        if self.score_proba:
            y_pred = self.estimator.predict_proba(X)
        else:
            y_pred = self.estimator.predict(X)
        loss_reference = self.loss(y_true=y, y_pred=y_pred)
        output_dict["loss_reference"] = loss_reference
        output_dict["loss_loco"] = dict()

        def _joblib_predict_one_group(estimator_j, X, y, j):
            """
            Compute the importance score for a single group of covariates
            removed.
            """
            X_minus_j = np.delete(X, self.groups[j], axis=1)

            if self.score_proba:
                y_pred_loco = estimator_j.predict_proba(X_minus_j)
            else:
                y_pred_loco = estimator_j.predict(X_minus_j)

            return y_pred_loco

        # Parallelize the computation of the importance scores for each group
        out_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_joblib_predict_one_group)(estimator_j, X, y, j)
            for j, estimator_j in enumerate(self._list_estimators)
        )

        return np.stack(out_list, axis=0)

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
        if len(self._list_estimators) == 0:
            raise ValueError("fit must be called before predict")
        for m in self._list_estimators:
            check_is_fitted(m)

        out_dict = dict()
        if self.score_proba:
            y_pred = self.estimator.predict_proba(X)
        else:
            y_pred = self.estimator.predict(X)

        loss_reference = self.loss(y_true=y, y_pred=y_pred)
        out_dict["loss_reference"] = loss_reference

        y_pred_loco = self.predict(X, y)

        out_dict["loss_loco"] = np.array(
            [self.loss(y_true=y, y_pred=y_pred_loco[j]) for j in range(self.n_groups)]
        )

        out_dict["importance"] = out_dict["loss_loco"] - loss_reference

        return out_dict
