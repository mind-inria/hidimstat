import numpy as np
from sklearn.utils import resample


########################## quantile aggregation method ##########################
def quantile_aggregation(pvals, gamma=0.05, adaptive=False):
    """
    Implements the quantile aggregation method for p-values.

    This method is based on :footcite:t:meinshausen2009pvalues.

    The function aggregates multiple p-values into a single p-value while controlling
    the family-wise error rate. It supports both fixed and adaptive quantile aggregation.

    Parameters
    ----------
    pvals : ndarray of shape (n_sampling*2, n_test)
        Matrix of p-values to aggregate. Each row represents a sampling instance
        and each column a hypothesis test.
    gamma : float, default=0.05
        Quantile level for aggregation. Must be in range (0,1].
    adaptive : bool, default=False
        If True, uses adaptive quantile aggregation which optimizes over multiple gamma values.
        If False, uses fixed quantile aggregation with the provided gamma value.

    Returns
    -------
    ndarray of shape (n_test,)
        Vector of aggregated p-values, one for each hypothesis test.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    The aggregated p-values are guaranteed to be valid p-values in [0,1].
    When adaptive=True, gamma is treated as the minimum gamma value to consider.
    """
    # if pvalues are one-dimensional, do nothing
    if pvals.shape[0] == 1:
        return pvals[0]
    if adaptive:
        return adaptive_quantile_aggregation(pvals, gamma)
    else:
        return fixed_quantile_aggregation(pvals, gamma)


def fixed_quantile_aggregation(pvals, gamma=0.5):
    """
    Quantile aggregation function

    For more details, see footcite:t:meinshausen2009pvalues

    Parameters
    ----------
    pvals : 2D ndarray (n_sampling*2, n_test)
        p-value

    gamma : float
        Percentile value used for aggregation.

    Returns
    -------
    pvalue aggregate: 1D ndarray (n_tests, )
        Vector of aggregated p-values

    References
    ----------
    .. footbibliography::
    """
    assert gamma > 0 and gamma <= 1, "gamma should be between 0 and 1"
    # equation 2.2 of meinshausen2009p
    converted_score = np.quantile(pvals, q=gamma, axis=0) / gamma
    return np.minimum(1, converted_score)


def adaptive_quantile_aggregation(pvals, gamma_min=0.05):
    """
    Adaptive version of quantile aggregation method

    For more details, see footcite:t:meinshausen2009pvalues

    Parameters
    ----------
    pvals : 2D ndarray (n_sampling*2, n_test)
        p-value
    gamma_min : float, default=0.05
        Minimum percentile value for adaptive aggregation

    Returns
    -------
    pvalue aggregate: 1D ndarray (n_tests, )
        Vector of aggregated p-values

    References
    ----------
    .. footbibliography::
    """
    assert gamma_min > 0 and gamma_min <= 1, "gamma min should between 0 and 1"

    n_iter, n_features = pvals.shape

    n_min = int(np.floor(gamma_min * n_iter))
    ordered_pval = np.sort(pvals, axis=0)[n_min:]
    # calculation of the pvalue / quantile (=j/m)
    # see equation 2.2 of `meinshausen2009p`
    P = (
        np.min(ordered_pval / np.arange(n_min, n_iter, 1).reshape(-1, 1), axis=0)
        * n_iter
    )
    # see equation 2.3 of `meinshausen2009p`
    pval_aggregate = np.minimum(1, (1 - np.log(gamma_min)) * P)
    return pval_aggregate


########################## False Discovery Proportion ##########################
def cal_fdp_power(selected, non_zero_index):
    """
    Calculate power and False Discovery Proportion

    Parameters
    ----------
    selected: list index (in R format) of selected non-null variables
    non_zero_index: true index of non-null variables

    Returns
    -------
    fdp: False Discoveries Proportion
    power: percentage of correctly selected variables over total number of
        non-null variables

    """
    # selected is the index list in R and will be different from index of
    # python by 1 unit

    if selected.size == 0:
        return 0.0, 0.0

    n_positives = len(non_zero_index)

    true_positive = np.intersect1d(selected, non_zero_index)
    false_positive = np.setdiff1d(selected, true_positive)

    fdp = len(false_positive) / max(1, len(selected))
    power = min(len(true_positive), n_positives) / n_positives

    return fdp, power


def fdr_threshold(pvals, fdr=0.1, method="bhq", reshaping_function=None):
    """
    False Discovery Rate thresholding method

    Parameters
    ----------
    pvals : 1D ndarray
        set of p-values
    fdr : float, default=0.1
        False Discovery Rate
    method : str, default='bhq'
        Method to control FDR.
        Available methods are:
        * 'bhq': Standard Benjamini-Hochberg :footcite:`benjamini1995controlling,bhy_2001`
        * 'bhy': Benjamini-Hochberg-Yekutieli :footcite:p:`bhy_2001`
        * 'ebh': e-Benjamini-Hochberg :footcite:`wang2022false`
    reshaping_function : function, default=None
        Reshaping function for Benjamini-Hochberg-Yekutieli method

    Returns
    -------
    threshold : float
        Threshold value

    References
    ----------
    .. footbibliography::
    """
    if method == "bhq":
        threshold = _bhq_threshold(pvals, fdr=fdr)
    elif method == "bhy":
        threshold = _bhy_threshold(
            pvals, fdr=fdr, reshaping_function=reshaping_function
        )
    elif method == "ebh":
        threshold = _ebh_threshold(pvals, fdr=fdr)
    else:
        raise ValueError("{} is not support FDR control method".format(method))
    return threshold


def _bhq_threshold(pvals, fdr=0.1):
    """
    Standard Benjamini-Hochberg :footcite:`benjamini1995controlling,bhy_2001`
    for controlling False discovery rate

    Parameters
    ----------
    pvals : 1D ndarray
        set of p-values
    fdr : float, default=0.1
        False Discovery Rate

    Returns
    -------
    threshold : float
        Threshold value

    References
    ----------
    .. footbibliography::
    """
    n_features = len(pvals)
    pvals_sorted = np.sort(pvals)
    selected_index = 2 * n_features
    for i in range(n_features - 1, -1, -1):
        if pvals_sorted[i] <= fdr * (i + 1) / n_features:
            selected_index = i
            break
    if selected_index <= n_features:
        threshold = pvals_sorted[selected_index]
    else:
        threshold = -1.0
    return threshold


def _ebh_threshold(evals, fdr=0.1):
    """
    e-BH procedure for FDR control :footcite:`wang2022false`

    Parameters
    ----------
    evals : 1D ndarray
        p-value
    fdr : float, default=0.1
        False Discovery Rate

    Returns
    -------
    threshold : float
        Threshold value

    References
    ----------
    .. footbibliography::
    """
    n_features = len(evals)
    evals_sorted = -np.sort(-evals)  # sort in descending order
    selected_index = 2 * n_features
    for i in range(n_features - 1, -1, -1):
        if evals_sorted[i] >= n_features / (fdr * (i + 1)):
            selected_index = i
            break
    if selected_index <= n_features:
        threshold = evals_sorted[selected_index]
    else:
        threshold = np.inf
    return threshold


def _bhy_threshold(pvals, reshaping_function=None, fdr=0.1):
    """
    Benjamini-Hochberg-Yekutieli :footcite:p:`bhy_2001` procedure for
    controlling FDR, with input shape function :footcite:p:`ramdas2017online`.

    Parameters
    ----------
    pvals : 1D ndarray
        set of p-values
    reshaping_function : function, default=None
        Reshaping function for Benjamini-Hochberg-Yekutieli method
    fdr : float, default=0.1
        False Discovery Rate

    Returns
    -------
    threshold : float
        Threshold value

    References
    ----------
    .. footbibliography::
    """
    n_features = len(pvals)
    pvals_sorted = np.sort(pvals)
    selected_index = 2 * n_features
    # Default value for reshaping function -- defined in
    # Benjamini & Yekutieli (2001)
    if reshaping_function is None:
        temp = np.arange(n_features)
        sum_inverse = np.sum(1 / (temp + 1))
        threshold = _bhq_threshold(pvals, fdr / sum_inverse)
    else:
        for i in range(n_features - 1, -1, -1):
            if pvals_sorted[i] <= fdr * reshaping_function(i + 1) / n_features:
                selected_index = i
                break
        if selected_index <= n_features:
            threshold = pvals_sorted[selected_index]
        else:
            threshold = -1.0
    return threshold


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


################# function for boostraping data ################################


def _subsampling(n_samples, train_size, groups=None, seed=0):
    """
    Random subsampling for statistical inference.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the dataset.
    train_size : float
        Fraction of samples to include in the training set (between 0 and 1).
    groups : ndarray, shape (n_samples,), optional (default=None)
        Group labels for samples.
        If not None, a subset of groups is selected.
    seed : int, optional (default=0)
        Random seed for reproducibility.

    Returns
    -------
    train_index : ndarray
        Indices of selected samples for training.
    """
    index_row = np.arange(n_samples) if groups is None else np.unique(groups)
    train_index = resample(
        index_row,
        n_samples=int(len(index_row) * train_size),
        replace=False,
        random_state=seed,
    )
    if groups is not None:
        train_index = np.arange(n_samples)[np.isin(groups, train_index)]
    return train_index
