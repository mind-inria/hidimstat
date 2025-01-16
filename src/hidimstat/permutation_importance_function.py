import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import root_mean_squared_error
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state
from sklearn.base import clone

from hidimstat.utils import _check_vim_predict_method


def permutation_importance(
    X,
    y,
    estimator,
    n_permutations: int = 50,
    loss: callable = root_mean_squared_error,
    method: str = "predict",
    random_state: int = None,
    n_jobs: int = 1,
    groups=None,
):
    """
    Parameters
    ----------
    X: np.ndarray of shape (n_samples, n_features)
        The input samples. Not used here.
    y: np.ndarray of shape (n_samples,)
        The target values. Not used here.
    estimator: scikit-learn compatible estimator
        The predictive model.
    n_permutations: int, default=50 / scikit-learn:n_repeats
        Number of permutations to perform.
    loss: callable, default=root_mean_squared_error / scikit-learn:scoring
        Loss function to evaluate the model performance.
    method: str, default='predict'
        Method to use for predicting values that will be used to compute
        the loss and the importance scores. The method must be implemented by the
        estimator. Supported methods are 'predict', 'predict_proba',
        'decision_function' and 'transform'.
    random_state: int, default=None
        Random seed for the permutation.
    n_jobs: int, default=1
        Number of jobs to run in parallel.
    groups: dict, default=None
        Dictionary of groups for the covariates. The keys are the group names
        and the values are lists of covariate indices.

    Return
    ------
    importance: array
        the importance scores for each group.

    References
    ----------
    .. footbibliography::
    """

    # check parameters
    _check_vim_predict_method(method)

    # define a random generator
    check_random_state(random_state)
    rng = np.random.RandomState(random_state)

    # management of the group
    if groups is None:
        n_groups = X.shape[1]
        groups_ = {j: [j] for j in range(n_groups)}
    else:
        n_groups = len(groups)
        if type(list(groups.values())[0][0]) is str:
            groups_ = {}
            for key, indexe_names in zip(groups.keys(), groups.values()):
                groups_[key] = []
                for index_name in indexe_names:
                    index = np.where(index_name == X.columns)[0]
                    assert len(index) == 1
                    groups_[key].append(index)
        else:
            groups_ = groups

    X_ = np.asarray(X)  # avoid the management of panda dataframe

    # reference loss
    try:
        y_pred = getattr(estimator, method)(X)
        estimator_ = estimator
    except NotFittedError:
        estimator_ = clone(estimator)
        # case for not fitted esimator
        estimator_.fit(X_, y)
        y_pred = getattr(estimator_, method)(X)
    loss_reference = loss(y, y_pred)

    def _predict_one_group(
        estimator,
        X,
        loss_reference,
        group_ids,
    ):
        """
        Compute the difference importance score for a single group of covariates.
        """
        # get ids
        non_group_ids = np.delete(np.arange(X.shape[1]), group_ids)
        # get data
        X_j = X[:, group_ids].copy()
        X_minus_j = np.delete(X, group_ids, axis=1)

        # Create an array X_perm_j of shape (n_permutations, n_samples, n_features)
        # where the j-th group of covariates is permuted
        X_perm_j = np.empty((n_permutations, X.shape[0], X.shape[1]))
        X_perm_j[:, :, non_group_ids] = X_minus_j

        # Create the permuted data for the j-th group of covariates
        group_j_permuted = np.array(
            [rng.permutation(X_j) for _ in range(n_permutations)]
        )
        X_perm_j[:, :, group_ids] = group_j_permuted

        # Reshape X_perm_j to allow for batch prediction
        X_perm_batch = X_perm_j.reshape(-1, X.shape[1])
        y_pred_perm = getattr(estimator, method)(X_perm_batch)

        # In case of classification, the output is a 2D array. Reshape accordingly
        if y_pred_perm.ndim == 1:
            y_pred_perm = y_pred_perm.reshape(n_permutations, X.shape[0])
        else:
            y_pred_perm = y_pred_perm.reshape(
                n_permutations, X.shape[0], y_pred_perm.shape[1]
            )
        diff_importance = np.array(
            [loss(y, y_pred_perm[i]) - loss_reference for i in range(n_permutations)]
        )
        return diff_importance

    # Parallelize the computation of the importance scores for each group
    # loss for all permutation
    list_diff_importance = Parallel(n_jobs=n_jobs)(
        delayed(_predict_one_group)(estimator_, X_, loss_reference, groups_[j])
        for j in groups_.keys()
    )

    # compute the importance
    importance = np.mean(list_diff_importance, axis=1)

    return importance
