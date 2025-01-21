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
    n_jobs: int = None,
    groups=None,
):
    """
    # Permutation importance
    
    Calculate permutation importance scores for features or feature groups in a machine learning model.
    Permutation importance is a model inspection technique that measures the increase in the model's
    prediction error after permuting a feature's values. A feature is considered "important" if shuffling
    its values increases the model error, because the model relied on the feature for the prediction.
    The implementation follows the methodology described in chapter 10 :cite:breimanRandomForests2001.
    One implementation: https://github.com/SkadiEye/deepTL/blob/master/R/4-2-permfit.R
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Training data. Can be numpy array or pandas DataFrame.
    y : np.ndarray of shape (n_samples,)
        Target values for the model.
    estimator : object
        A fitted estimator object implementing scikit-learn estimator interface.
        The estimator must have a fitting method and one of the following prediction methods:
        'predict', 'predict_proba', 'decision_function', or 'transform'.
    n_permutations : int, default=50
        Number of times to permute each feature or feature group.
        Higher values give more stable results but take longer to compute.
    loss : callable, default=root_mean_squared_error
        Function to measure the prediction error. Must take two arguments (y_true, y_pred)
        and return a scalar value. Higher return values must indicate worse predictions.
    method : str, default='predict'
        The estimator method used for prediction. Must be one of:
        - 'predict': Use estimator.predict()
        - 'predict_proba': Use estimator.predict_proba()
        - 'decision_function': Use estimator.decision_function()
        - 'transform': Use estimator.transform()
    random_state : int, default=None
        Controls the randomness of the feature permutations.
        Pass an int for reproducible results across multiple function calls.
    n_jobs : int, default=None
        Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors.
    groups : dict, default=None
        Dictionary specifying feature groups. Keys are group names and values are lists of feature
        indices or feature names (if X is a pandas DataFrame). If None, each feature is treated
        as its own group.
    
    Returns
    -------
    importance : np.ndarray of shape (n_features,) or (n_groups,)
        The importance scores for each feature or feature group.
        Higher values indicate more important features.
    list_loss_j : np.ndarray
        Array containing all computed loss values for each permutation of each feature/group.
    loss_reference : float
        The reference loss (baseline) computed on the original, non-permuted data.
   
    Notes
    -----
    The implementation supports both individual feature importance and group feature importance.
    For group importance, features within the same group are permuted together.

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

    # compute the reference residual
    try:
        y_pred = getattr(estimator, method)(X)
        estimator_ = estimator
    except NotFittedError:
        estimator_ = clone(estimator)
        # case for not fitted esimator
        estimator_.fit(X_, y)
        y_pred = getattr(estimator_, method)(X)
    loss_reference = loss(y, y_pred)

    # Parallelize the computation of the residual for each permutation
    # of each group
    list_loss_j = Parallel(n_jobs=n_jobs)(
        delayed(_predict_one_group)(
            estimator_,
            groups_[j],
            X_,
            y,
            loss,
            n_permutations,
            rng,
            method,
        )
        for j in groups_.keys()
    )
    list_loss_j = np.array(list_loss_j)

    # compute the importance
    # equation 5 of mi2021permutation
    importance = np.mean(list_loss_j - loss_reference, axis=1) 

    return importance, list_loss_j, loss_reference


def _predict_one_group(
    estimator, group_ids, X, y, loss, n_permutations, rng, method
):
    """
    Compute prediction loss scores after permuting a single group of features.

    Parameters
    ----------
    estimator : object
        Fitted estimator implementing scikit-learn API
    group_ids : list
        Indices of features in the group to permute
    X : np.ndarray
        Input data matrix
    y : np.ndarray 
        Target values
    loss : callable
        Loss function to evaluate predictions
    n_permutations : int
        Number of permutations to perform
    rng : RandomState
        Random number generator instance
    method : str
        Prediction method to use ('predict', 'predict_proba', etc.)

    Returns
    -------
    list
        Loss values for each permutation
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
    group_j_permuted = np.array([rng.permutation(X_j) for _ in range(n_permutations)])
    X_perm_j[:, :, group_ids] = group_j_permuted

    # Reshape X_perm_j to allow for remove the indexation by groups
    X_perm_batch = X_perm_j.reshape(-1, X.shape[1])
    y_pred_perm = getattr(estimator, method)(X_perm_batch)

    if y_pred_perm.ndim == 1:
        # one value per y: regression
        y_pred_perm = y_pred_perm.reshape(n_permutations, X.shape[0])
    else:
        # probability per y: classification
        y_pred_perm = y_pred_perm.reshape(
            n_permutations, X.shape[0], y_pred_perm.shape[1]
        )
    loss_i = [loss(y, y_pred_perm[i]) for i in range(n_permutations)]
    return loss_i
