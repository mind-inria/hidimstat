import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.metrics import root_mean_squared_error


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
        n_permutations: int = 50,
        loss: callable = root_mean_squared_error,
        score_proba: bool = False,
        random_state: int = None,
        n_jobs: int = 1,
    ):

        check_is_fitted(estimator)
        self.estimator = estimator
        self.n_permutations = n_permutations

        self.random_state = random_state
        self.loss = loss
        self.score_proba = score_proba
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

    def predict(self, X, y):
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
        if self.score_proba:
            y_pred = self.estimator.predict_proba(X)
        else:
            y_pred = self.estimator.predict(X)
        loss_reference = self.loss(y_true=y, y_pred=y_pred)

        def _joblib_predict_one_group(X, j):
            """
            Compute the importance score for a single group of covariates.
            """
            list_y_pred_perm = []
            X_j = X[:, self.groups[j]].copy()
            X_minus_j = np.delete(X, self.groups[j], axis=1)
            group_ids = self.groups[j]
            non_group_ids = np.delete(np.arange(X.shape[1]), group_ids)

            for _ in range(self.n_permutations):
                X_j_perm = self.rng.permutation(X_j)
                X_perm = np.empty_like(X)
                X_perm[:, non_group_ids] = X_minus_j
                X_perm[:, group_ids] = X_j_perm

                if self.score_proba:
                    y_pred_perm = self.estimator.predict_proba(X_perm)
                else:
                    y_pred_perm = self.estimator.predict(X_perm)

                list_y_pred_perm.append(y_pred_perm)
            return np.array(list_y_pred_perm)

        # Parallelize the computation of the importance scores for each group
        out_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_joblib_predict_one_group)(X, j) for j in range(len(self.groups))
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
        if self.score_proba:
            y_pred = self.estimator.predict_proba(X)
        else:
            y_pred = self.estimator.predict(X)
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
