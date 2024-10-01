import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from sklearn.metrics import mean_squared_error


class PermutationImportance(BaseEstimator, TransformerMixin):
    """


    Parameters
    ----------
    estimator: scikit-learn compatible estimator
        The predictive model.
    n_perm: int, default=50
        Number of permutations to perform.
    groups: dict, default=None
        Dictionary of groups for the covariates. The keys are the group names
        and the values are lists of covariate indices.
    loss: callable, default=mean_squared_error
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

    def __init__(self,
                 estimator,
                 n_perm: int = 50,
                 groups: dict = None,
                 loss: callable = mean_squared_error,
                 score_proba: bool = False,
                 random_state: int = None,
                 n_jobs: int = 1
                 ):

        check_is_fitted(estimator)
        self.estimator = estimator
        self.n_perm = n_perm
        self.groups = groups
        self.random_state = random_state
        self.loss = loss
        self.score_proba = score_proba
        self.n_jobs = n_jobs
        self.rng = np.random.RandomState(random_state)
        self.nb_groups = None

    def fit(self, X, y):
        return self

    def predict(self, X, y):
        """
        Compute the Permutation importance scores for each group of covariates.

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
            - 'loss_perm': a dictionary containing the loss of the model with
            the permuted data for each group.
            - 'importance': the importance scores for each group.
        """
        check_is_fitted(self.estimator)
        if self.groups is None:
            self.nb_groups = X.shape[1]
            self.groups = {j: [j] for j in range(self.nb_groups)}
        else:
            self.nb_groups = len(self.groups)
        output_dict = dict()
        if self.score_proba:
            y_pred = self.estimator.predict_proba(X)
        else:
            y_pred = self.estimator.predict(X)
        loss_reference = self.loss(y_true=y, y_pred=y_pred)
        output_dict["loss_reference"] = loss_reference
        output_dict['loss_perm'] = dict()

        def joblib_predict_one_gp(estimator, X, y, j):
            """
            Compute the importance score for a single group of covariates.
            """
            list_loss_perm = []
            X_j = X[:, self.groups[j]].copy()
            X_minus_j = np.delete(X, self.groups[j], axis=1)
            group_ids = self.groups[j]
            non_group_ids = np.delete(np.arange(X.shape[1]), group_ids)

            for _ in range(self.n_perm):
                X_j_perm = self.rng.permutation(X_j)
                X_perm = np.empty_like(X)
                X_perm[:, non_group_ids] = X_minus_j
                X_perm[:, group_ids] = X_j_perm

                if self.score_proba:
                    y_pred_perm = self.estimator.predict_proba(X_perm)
                else:
                    y_pred_perm = self.estimator.predict(X_perm)
                list_loss_perm.append(self.loss(y_true=y, y_pred=y_pred_perm))
            return np.array(list_loss_perm)

        # Parallelize the computation of the importance scores for each group
        out_list = Parallel(n_jobs=self.n_jobs)(
            delayed(joblib_predict_one_gp)(self.estimator, X, y, j)
            for j in range(len(self.groups)))

        for j, list_loss_perm in enumerate(out_list):
            output_dict['loss_perm'][j] = list_loss_perm

        output_dict['importance'] = np.array([
            np.mean(
                output_dict['loss_perm'][j] - output_dict['loss_reference'])
            for j in range(self.nb_groups)])

        return output_dict
