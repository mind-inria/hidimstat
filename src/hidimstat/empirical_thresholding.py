import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR


def empirical_thresholding(
    X,
    y,
    linear_estimator=GridSearchCV(
        LinearSVR(), param_grid={"C": np.logspace(-7, 1, 9)}, n_jobs=None
    ),
):
    """
    Perform empirical thresholding on the input data and target using a linear
    estimator.

    This function fits a linear estimator to the input data and target, 
    and then uses the estimated coefficients to perform empirical thresholding.
    The threshold is calculated for keeping only extreme coefficients.
    For more details, see the section 6.3.2 of :cite:`chevalier_statistical_2020`

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        The input data.
    y : ndarray, shape (n_samples,)
        The target values.
    linear_estimator : estimator object, optional (default=GridSearchCV(
            LinearSVR(),param_grid={"C": np.logspace(-7, 1, 9)}, n_jobs=None))
        The linear estimator to use for thresholding. It should be a scikit-learn
        estimator object that implements the `fit` method and has a `coef_` 
        attribute or a `best_estimator_` attribute with a `coef_` attribute 
        (e.g., a `GridSearchCV` object).

    Returns
    -------
    beta_hat : ndarray, shape (n_features,)
        The estimated coefficients of the linear estimator.
    scale : ndarray, shape (n_features,)
        The threshold values for each feature.

    Raises
    ------
    ValueError
        If the `linear_estimator` does not have a `coef_` attribute 
        or a `best_estimator_` attribute with a `coef_` attribute.

    Notes
    -----
    The threshold is calculated as the standard deviation of the estimated 
    coefficients multiplied by the square root of the number of features. 
    This is based on the assumption that the coefficients follow a normal
    distribution with mean zero.
    """
    _, n_features = X.shape

    linear_estimator.fit(X, y)

    if hasattr(linear_estimator, "coef_"):
        beta_hat = linear_estimator.coef_
    elif hasattr(linear_estimator, "best_estimator_") and hasattr(
        linear_estimator.best_estimator_, "coef_"
    ):
        beta_hat = linear_estimator.best_estimator_.coef_  # for CV object
    else:
        raise ValueError("linear estimator should be linear.")

    std = norm(beta_hat) / np.sqrt(n_features)
    scale = std * np.ones(beta_hat.size)

    return beta_hat, scale
