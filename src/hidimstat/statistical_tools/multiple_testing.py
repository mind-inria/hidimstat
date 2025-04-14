import numpy as np


def cal_fdp_power(selected, non_zero_index):
    """
    Calculate False Discovery Proportion and statistical power

    Parameters
    ----------
    selected : ndarray
        Array of indices of selected variables (R-style indexing)
    non_zero_index : ndarray
        Array of true non-null variable indices

    Returns
    -------
    fdp : float
        False Discovery Proportion (number of false discoveries / total discoveries)
    power : float 
        Statistical power (number of true discoveries / number of non-null variables)
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
    Calculate threshold for False Discovery Rate control methods.

    Parameters
    ----------
    pvals : 1D ndarray
        Set of p-values to threshold
    fdr : float, default=0.1
        Target False Discovery Rate level
    method : {'bhq', 'bhy', 'ebh'}, default='bhq'
        Method for FDR control:
        * 'bhq': Standard Benjamini-Hochberg procedure
        * 'bhy': Benjamini-Hochberg-Yekutieli procedure
        * 'ebh': e-Benjamini-Hochberg procedure
    reshaping_function : callable
        Reshaping function for BHY method, default uses sum of reciprocals

    Returns
    -------
    threshold : float
        Threshold value for p-values. P-values below this threshold are rejected.

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
    Standard Benjamini-Hochberg 
    for controlling False discovery rate

    Calculate threshold for standard Benjamini-Hochberg procedure 
    :footcite:`benjamini1995controlling,bhy_2001` for False Discovery Rate (FDR)
    control.

    Parameters
    ----------
    pvals : 1D ndarray
        Array of p-values to threshold
    fdr : float, default=0.1
        Target False Discovery Rate level

    Returns
    -------
    threshold : float
        Threshold value for p-values. P-values below this threshold are rejected.

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
        Array of e-values to threshold
    fdr : float, default=0.1
        Target False Discovery Rate level

    Returns
    -------
    threshold : float
        Threshold value for e-values. E-values above this threshold are rejected.

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
        Benjamini-Hochberg-Yekutieli  procedure for
    controlling FDR

    Calculate threshold for Benjamini-Hochberg-Yekutieli procedure
    :footcite:p:`bhy_2001` for False Discovery Rate control, 
    with input shape function :footcite:p:`ramdas2017online`.

    Parameters
    ----------
    pvals : 1D ndarray
        Array of p-values to threshold
    reshaping_function : callable, default=None
        Function to reshape FDR threshold. If None, uses sum of reciprocals.
    fdr : float, default=0.1
        Target False Discovery Rate level

    Returns
    -------
    threshold : float
        Threshold value for p-values. P-values below this threshold are rejected.

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
