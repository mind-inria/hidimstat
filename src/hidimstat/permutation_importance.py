import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.utils import check_random_state

from hidimstat.base_perturbation import BasePerturbation


class PermutationImportance(BasePerturbation):
    def __init__(
        self,
        estimator,
        loss: callable = root_mean_squared_error,
        method: str = "predict",
        n_jobs: int = 1,
        n_permutations: int = 50,
        random_state: int = None,
    ):
        """
        Permutation Importance algorithm as presented in
        :footcite:t:`breimanRandomForests2001`. For each variable/group of variables,
        the importance is computed as the difference between the loss of the initial
        model and the loss of the model with the variable/group permuted.

        Parameters
        ----------
        estimator : sklearn compatible estimator, optionals
            The estimator to use for the prediction.
        loss : callable, default=root_mean_squared_error
            The loss function to use when comparing the perturbed model to the full
            model.
        method : str, default="predict"
            The method to use for the prediction. This determines the predictions passed
            to the loss function. Supported methods are "predict", "predict_proba",
            "decision_function", "transform".
        n_jobs : int, default=1
            The number of jobs to run in parallel. Parallelization is done over the
            variables or groups of variables.
        n_permutations : int, default=50
            The number of permutations to perform. For each variable/group of variables,
            the mean of the losses over the `n_permutations` is computed.
        random_state : int, default=None
            The random state to use for sampling.

        References
        ----------
        .. footbibliography::
        """
        super().__init__(
            estimator=estimator,
            loss=loss,
            method=method,
            n_jobs=n_jobs,
            n_permutations=n_permutations,
        )
        self.rng = check_random_state(random_state)

    def _permutation(self, X, group_id):
        """Create the permuted data for the j-th group of covariates"""
        X_perm_j = np.array(
            [
                self.rng.permutation(X[:, self._groups_ids[group_id]].copy())
                for _ in range(self.n_permutations)
            ]
        )
        return X_perm_j
