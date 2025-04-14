import numpy as np


########################## alpha Max Calculation ##########################
def _alpha_max(X, y, use_noise_estimate=False, fill_diagonal=False, axis=None):
    """
    Calculate alpha_max, which is the smallest value of the regularization parameter
    in the LASSO regression that yields non-zero coefficients.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data
    y : array-like of shape (n_samples,) or (n_samples, n_features)
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

    Xt_y = np.dot(X.T, y)
    if fill_diagonal:
        np.fill_diagonal(Xt_y, 0)

    alpha_max = np.max(Xt_y, axis=axis) / n_samples

    if use_noise_estimate:
        # estimate the noise level
        norm_y = np.linalg.norm(y, ord=2)
        sigma_star = norm_y / np.sqrt(n_samples)
        # rectified by the noise
        alpha_max = np.abs(alpha_max) / sigma_star

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
