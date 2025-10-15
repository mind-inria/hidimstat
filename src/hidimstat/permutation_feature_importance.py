import numpy as np
from sklearn.metrics import root_mean_squared_error

from hidimstat._utils.utils import check_random_state
from hidimstat.base_perturbation import BasePerturbation


class PFI(BasePerturbation):
    def __init__(
        self,
        estimator,
        loss=root_mean_squared_error,
        method: str = "predict",
        n_permutations: int = 50,
        features_groups=None,
        random_state: int = None,
        n_jobs: int = 1,
    ):
        """
        Permutation Feature Importance algorithm as presented in
        :footcite:t:`breimanRandomForests2001`. For each feature/group of features,
        the importance is computed as the difference between the loss of the initial
        model and the loss of the model with the feature/group permuted.
        The method was also used in :footcite:t:`mi2021permutation`

        Parameters
        ----------
        estimator : sklearn compatible estimator, optionals
            The estimator to use for the prediction.
        loss : callable, default=root_mean_squared_error
            The loss function to use when comparing the perturbed model to the full
            model.
        method : str, default="predict"
            The method to use for the prediction. This determines the predictions passed
            to the loss function. Supported methods are "predict", "predict_proba" or
            "decision_function".
        n_permutations : int, default=50
            The number of permutations to perform. For each feature/group of features,
            the mean of the losses over the `n_permutations` is computed.
        features_groups: dict or None,  default=None
            A dictionary where the keys are the group names and the values are the
            list of column names corresponding to each features group. If None,
            the features_groups are identified based on the columns of X.
        random_state : int, default=None
            The random state to use for sampling.
        n_jobs : int, default=1
            The number of jobs to run in parallel. Parallelization is done over the
            features or groups of features.

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
            features_groups=features_groups,
            random_state=random_state,
        )

    def _permutation(self, X, features_group_id, random_state=None):
        """Create the permuted data for the j-th group of covariates"""
        rng = check_random_state(random_state)
        X_perm_j = np.array(
            [
                rng.permutation(
                    X[:, self._features_groups_ids[features_group_id]].copy()
                )
                for _ in range(self.n_permutations)
            ]
        )
        return X_perm_j
