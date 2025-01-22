import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import root_mean_squared_error
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state
from sklearn.base import clone

from hidimstat.utils import _check_vim_predict_method


def _base_permutation_importance(
    X,
    y,
    estimator,
    n_permutations: int = 50,
    loss: callable = root_mean_squared_error,
    method: str = "predict",
    n_jobs: int = None,
    groups=None,
    permutation_data=None,
    update_estimator=False
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
                    groups_[key].append(index[0])
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
    if permutation_data is None:
        raise ValueError("Require a function")
    list_result = Parallel(n_jobs=n_jobs)(
        delayed(_predict_one_group_generic)(
            j,
            estimator_,
            groups_[j],
            X_,
            y,
            loss,
            n_permutations,
            method,
            permutation_data=permutation_data,
            update_estimator=update_estimator
        )
        for j in groups_.keys()
    )
    list_loss_j = np.array([i[0] for i in list_result])
    list_additional_output = [i[1] for i in list_result]

    # compute the importance
    # equation 5 of mi2021permutation
    importance = np.mean(list_loss_j - loss_reference, axis=1)

    return (importance, list_loss_j, loss_reference), list_additional_output


def _predict_one_group_generic(
    index_group, estimator, group_ids, X, y, loss, n_permutations, method, permutation_data=None,
    update_estimator=False
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

    if permutation_data is None:
        raise ValueError("require a function")
    else:
        X_perm_j, additional_output = permutation_data(index_group=index_group, 
            X_minus_j=X_minus_j, X_j=X_j, X_perm_j=X_perm_j, group_ids=group_ids
        )
    if update_estimator:
        estimator = additional_output[0]
        additional_output = additional_output[1]

    # Reshape X_perm_j to allow for remove the indexation by groups
    y_pred_perm = getattr(estimator, method)(X_perm_j)

    if y_pred_perm.ndim == 1:
        # one value per y: regression
        y_pred_perm = y_pred_perm.reshape(n_permutations, X.shape[0])
    else:
        # probability per y: classification
        y_pred_perm = y_pred_perm.reshape(
            n_permutations, X.shape[0], y_pred_perm.shape[1]
        )
    loss_i = [loss(y, y_pred_perm[i]) for i in range(n_permutations)]
    return loss_i, additional_output


def permutation_importance(
    *args,
    # additional argument
    random_state: int = None,
    n_permutations: int = 50,
    **kwargs,
):
    # define a random generator
    check_random_state(random_state)
    rng = np.random.RandomState(random_state)

    def permute_column(index_group, X_minus_j, X_j, X_perm_j, group_ids):
        # Create the permuted data for the j-th group of covariates
        group_j_permuted = np.array(
            [rng.permutation(X_j) for _ in range(n_permutations)]
        )
        X_perm_j[:, :, group_ids] = group_j_permuted
        X_perm_j = X_perm_j.reshape(-1, X_minus_j.shape[1] + X_j.shape[1])
        return X_perm_j, None

    result, _ = _base_permutation_importance(
        *args, **kwargs, n_permutations=n_permutations, permutation_data=permute_column
    )
    return result


def loco(
    X_train,
    y_train,
    *args,
    # additional argument
    **kwargs,
):
    if len(args)>=3:
        estimator = clone(args[2])
    else:
        estimator = kwargs['estimator']
    X_train_ = np.asarray(X_train)
     

    def create_new_estimator(index_group, X_minus_j, X_j, X_perm_j, group_ids):
        # Modify the actual estimator for fitting without the colomn j
        X_train_minus_j = np.delete(X_train_, group_ids, axis=1)
        estimator_ = clone(estimator)
        estimator_.fit(X_train_minus_j, y_train)
        X_perm_j = X_minus_j
        return X_perm_j, (estimator_, estimator_)

    result, list_estimator = _base_permutation_importance(
        *args, **kwargs, n_permutations=1, permutation_data=create_new_estimator, update_estimator=True
    )
    return result


def cpi(
    X_train,
    *args,
    # additional argument
    imputation_model=None,
    imputation_method: str = "predict",
    random_state: int = None,
    distance_residual: callable = np.subtract,
    n_permutations: int = 50,
    **kwargs,
):
    X_train_ = np.asarray(X_train)
    if imputation_model is None:
        raise ValueError("missing estimator for imputation")
    n_permutations = n_permutations
    # define a random generator
    check_random_state(random_state)
    rng = np.random.RandomState(random_state)

    def permutation_conditional(index_group, X_minus_j, X_j, X_perm_j, group_ids):
        X_train_j = X_train_[:, group_ids].copy()
        X_train_minus_j = np.delete(X_train_, group_ids, axis=1)
        # create X from residual
        # add one parameter: estimator_imputation
        if type(imputation_model) is list or type(imputation_model) is dict:
            estimator_ = imputation_model[index_group]
        else:
            estimator_ = clone(imputation_model)
        estimator_.fit(X_train_minus_j, X_train_j)

        # Reshape X_perm_j to allow for remove the indexation by groups
        X_j_hat = getattr(estimator_, imputation_method)(X_minus_j)

        if X_j_hat.ndim == 1 or X_j_hat.shape[1] == 1:
            # one value per X_j_hat: regression
            X_j_hat = X_j_hat.reshape(X_j.shape)
        else:
            # probability per X_j_hat: classification
            X_j_hat = X_j_hat.reshape(X_j.shape[0], X_j_hat.shape[1])
        residual_j = distance_residual(X_j, X_j_hat)

        # Create the permuted data for the j-th group of covariates
        residual_j_perm = np.array(
            [rng.permutation(residual_j) for _ in range(n_permutations)]
        )
        X_perm_j[:, :, group_ids] = X_j_hat[np.newaxis, :, :] + residual_j_perm
        
        X_perm_j = X_perm_j.reshape(-1, X_minus_j.shape[1] + X_j.shape[1])
        
        return X_perm_j, estimator_

    result, list_estimator = _base_permutation_importance(
        *args,
        **kwargs,
        n_permutations=n_permutations,
        permutation_data=permutation_conditional,
    )
    return result
