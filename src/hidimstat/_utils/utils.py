import numpy as np


########################## alpha Max Calculation ##########################
def _alpha_max(X, y, use_noise_estimate=False):
    """
    Calculate alpha_max, which is the smallest value of the regularization parameter
    in the LASSO regression that yields non-zero coefficients.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data
    y : array-like of shape (n_samples,)
        Target values
    use_noise_estimate : bool, default=True
        Whether to use noise estimation in the calculation

    Returns
    -------
    float
        The maximum alpha value

    Notes
    -----
    For LASSO regression, any alpha value larger than alpha_max will result in
    all zero coefficients. This provides an upper bound for the regularization path.
    """
    n_samples, _ = X.shape

    if not use_noise_estimate:
        alpha_max = np.max(np.dot(X.T, y)) / n_samples
    else:
        # estimate the noise level
        norm_y = np.linalg.norm(y, ord=2)
        sigma_star = norm_y / np.sqrt(n_samples)

        alpha_max = np.max(np.abs(np.dot(X.T, y)) / (n_samples * sigma_star))

    return alpha_max


########################## function for using Sklearn ##########################
def _check_vim_predict_method(method):
    """
    Validates that the method is a valid scikit-learn prediction method for variable importance measures.

    Parameters
    ----------
    method : str
        The scikit-learn prediction method to validate.

    Returns
    -------
    str
        The validated method if valid.

    Raises
    ------
    ValueError
        If the method is not one of the standard scikit-learn prediction methods:
        'predict', 'predict_proba', 'decision_function', or 'transform'.
    """
    if method in ["predict", "predict_proba", "decision_function", "transform"]:
        return method
    else:
        raise ValueError(
            "The method {} is not a valid method for variable importance measure prediction".format(
                method
            )
        )
