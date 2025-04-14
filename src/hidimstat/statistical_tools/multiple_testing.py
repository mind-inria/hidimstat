import numpy as np


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
