import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.utils import check_random_state

from hidimstat.base_perturbation import BasePerturbation
from hidimstat._utils.docstring import _aggregate_docstring


class PFI(BasePerturbation):
    """
    Permutation Feature Importance algorithm

    This as presented in :footcite:t:`breimanRandomForests2001`.
    For each variable/group of variables, the importance is computed as
    the difference between the loss of the initial model and the loss of
    the model with the variable/group permuted.
    The method was also used in :footcite:t:`mi2021permutation`

    Parameters
    ----------
    estimator : sklearn compatible estimator, optionals
        The estimator to use for the prediction.
    method : str, default="predict"
        The method to use for the prediction. This determines the predictions passed
        to the loss function. Supported methods are "predict", "predict_proba" or
        "decision_function".
    loss : callable, default=root_mean_squared_error
        The loss function to use when comparing the perturbed model to the full
        model.
    n_permutations : int, default=50
        The number of permutations to perform. For each variable/group of variables,
        the mean of the losses over the `n_permutations` is computed.
    random_state : int, default=None
        The random state to use for sampling.
    n_jobs : int, default=1
        The number of jobs to run in parallel. Parallelization is done over the
        variables or groups of variables.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        estimator,
        method: str = "predict",
        loss: callable = root_mean_squared_error,
        n_permutations: int = 50,
        random_state: int = None,
        n_jobs: int = 1,
    ):

        super().__init__(
            estimator=estimator,
            method=method,
            loss=loss,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
        )
        self.random_state = random_state

    def _permutation(self, X, group_id):
        """Create the permuted data for the j-th group of covariates"""
        self.random_state = check_random_state(self.random_state)
        X_perm_j = np.array(
            [
                self.random_state.permutation(X[:, self._groups_ids[group_id]].copy())
                for _ in range(self.n_permutations)
            ]
        )
        return X_perm_j


def pfi(
    estimator,
    X,
    y,
    groups: dict = None,
    method: str = "predict",
    loss: callable = root_mean_squared_error,
    n_permutations: int = 50,
    k_best=None,
    percentile=None,
    threshold=None,
    threshold_pvalue=None,
    random_state: int = None,
    n_jobs: int = 1,
):
    methods = PFI(
        estimator=estimator,
        method=method,
        loss=loss,
        n_permutations=n_permutations,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    methods.fit_importance(
        X,
        y,
        groups=groups,
    )
    selection = methods.selection(
        k_best=k_best,
        percentile=percentile,
        threshold=threshold,
        threshold_pvalue=threshold_pvalue,
    )
    return selection, methods.importances_, methods.pvalues_


# use the docstring of the class for the function
pfi.__doc__ = _aggregate_docstring(
    [
        PFI.__doc__,
        PFI.__init__.__doc__,
        PFI.fit_importance.__doc__,
        PFI.selection.__doc__,
    ],
    """
    Returns
    -------
    selection : ndarray of shape (n_features,)
        Boolean array indicating selected features (True = selected)
    importances : ndarray of shape (n_features,)
        Feature importance scores/test statistics.
    pvalues : ndarray of shape (n_features,)
         P-values for importance scores.
    """,
)
