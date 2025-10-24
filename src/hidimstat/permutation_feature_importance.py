from functools import partial

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import mean_squared_error

from hidimstat._utils.docstring import _aggregate_docstring
from hidimstat._utils.utils import check_random_state
from hidimstat.base_perturbation import BasePerturbation


class PFI(BasePerturbation):
    """
    Permutation Feature Importance algorithm

    This as presented in :footcite:t:`breimanRandomForests2001`.
    For each variable/group of variables, the importance is computed as the
    difference between the loss of the initial model and the loss of the model
    with the variable/group permuted.
    The method was also used in :footcite:t:`mi2021permutation`

    Parameters
    ----------
    estimator : sklearn compatible estimator
        The estimator to use for the prediction.
    method : str, default="predict"
        The method to use for the prediction. This determines the predictions passed
        to the loss function. Supported methods are "predict", "predict_proba" or
        "decision_function".
    loss : callable, default=mean_squared_error
        The loss function to use when comparing the perturbed model to the full
        model.
    n_permutations : int, default=50
        The number of permutations to perform. For each variable/group of variables,
        the mean of the losses over the `n_permutations` is computed.
    statistical_test : callable or str, default="nb-ttest"
        Statistical test function for computing p-values of importance scores.
    features_groups: dict or None, default=None
        A dictionary where the keys are the group names and the values are the
        list of column names corresponding to each features group. If None,
        the features_groups are identified based on the columns of X.
    random_state : int or None, default=None
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
        loss: callable = mean_squared_error,
        n_permutations: int = 50,
        statistical_test="nb-ttest",
        features_groups=None,
        random_state: int = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            estimator=estimator,
            method=method,
            loss=loss,
            n_permutations=n_permutations,
            statistical_test=statistical_test,
            features_groups=features_groups,
            random_state=random_state,
            n_jobs=n_jobs,
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


def pfi(
    estimator,
    X,
    y,
    method: str = "predict",
    loss: callable = mean_squared_error,
    n_permutations: int = 50,
    test_statistic=partial(wilcoxon, axis=1),
    features_groups=None,
    k_best=None,
    percentile=None,
    threshold_min=None,
    threshold_max=None,
    random_state: int = None,
    n_jobs: int = 1,
):
    methods = PFI(
        estimator=estimator,
        method=method,
        loss=loss,
        n_permutations=n_permutations,
        statistical_test=test_statistic,
        features_groups=features_groups,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    methods.fit_importance(X, y)
    selection = methods.importance_selection(
        k_best=k_best,
        percentile=percentile,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
    )
    return selection, methods.importances_, methods.pvalues_


# use the docstring of the class for the function
pfi.__doc__ = _aggregate_docstring(
    [
        PFI.__doc__,
        PFI.__init__.__doc__,
        PFI.fit_importance.__doc__,
        PFI.importance_selection.__doc__,
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
