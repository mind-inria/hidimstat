import numpy as np


########################## quantile aggregation method ##########################
def quantile_aggregation(pvals, gamma=0.05, n_grid=20, adaptive=False):
    """
    Implements the quantile aggregation method for p-values based on :cite:meinshausen2009p.

    The function aggregates multiple p-values into a single p-value while controlling
    the family-wise error rate. It supports both fixed and adaptive quantile aggregation.

    Parameters
    ----------
    pvals : ndarray of shape (n_sampling*2, n_test)
        Matrix of p-values to aggregate. Each row represents a sampling instance
        and each column a hypothesis test.
    gamma : float, default=0.05
        Quantile level for aggregation. Must be in range (0,1].
    n_grid : int, default=20
        Number of grid points to use for adaptive aggregation. Only used if adaptive=True.
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
        return _adaptive_quantile_aggregation(pvals, gamma, n_grid=n_grid)
    else:
        return _fixed_quantile_aggregation(pvals, gamma)


def _fixed_quantile_aggregation(pvals, gamma=0.5):
    """
    Quantile aggregation function based on :cite:meinshausen2009p

    Parameters
    ----------
    pvals : 2D ndarray (n_sampling*2, n_test)
        p-value

    gamma : float
        Percentile value used for aggregation.

    Returns
    -------
    1D ndarray (n_tests, )
        Vector of aggregated p-values

    References
    ----------
    .. footbibliography::
    """
    assert gamma > 0 and gamma <= 1, "gamma should be between O and 1"
    # equation 2.2 of meinshausen2009p
    converted_score = (1 / gamma) * (np.percentile(pvals, q=100 * gamma, axis=0))
    return np.minimum(1, converted_score)


def _adaptive_quantile_aggregation(pvals, gamma_min=0.05, n_grid=20):
    """
    Adaptive version of quantile aggregation method based on :cite:meinshausen2009p

    Parameters
    ----------
    pvals : 2D ndarray (n_sampling*2, n_test)
        p-value
    gamma_min : float, default=0.05
        Minimum percentile value for adaptive aggregation

    Returns
    -------
    1D ndarray (n_tests, )
        Vector of aggregated p-values

    References
    ----------
    .. footbibliography::
    """

    gammas = np.linspace(gamma_min, 1.0, n_grid)
    list_quantiles = np.array(
        [_fixed_quantile_aggregation(pvals, gamma) for gamma in gammas]
    )
    # equation 2.3 of meinshausen2009p
    return np.minimum(1, (1 - np.log(gamma_min)) * list_quantiles.min(0))


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
        return pvals_sorted[selected_index]
    else:
        # no threshold, all the pvalue are positive
        return -1.0


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
        return evals_sorted[selected_index]
    else:
        # no threshold, no e-value is significant at the chosen level
        return np.inf


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
        return _bhq_threshold(pvals, fdr / sum_inverse)
    else:
        for i in range(n_features - 1, -1, -1):
            if pvals_sorted[i] <= fdr * reshaping_function(i + 1) / n_features:
                selected_index = i
                break
        if selected_index <= n_features:
            return pvals_sorted[selected_index]
        else:
            # no threshold, all the p-values are positive
            return -1.0


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