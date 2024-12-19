import numpy as np


def quantile_aggregation(pvals, gamma=0.5, gamma_min=0.05, adaptive=False):
    """
    This function implements the quantile aggregation method for p-values.
    """
    # if pvalues are one-dimensional, do nothing
    if pvals.shape[0] == 1:
        return pvals[0]
    if adaptive:
        return _adaptive_quantile_aggregation(pvals, gamma_min)
    else:
        return _fixed_quantile_aggregation(pvals, gamma)


def fdr_threshold(pvals, fdr=0.1, method="bhq", reshaping_function=None):
    if method == "bhq":
        return _bhq_threshold(pvals, fdr=fdr)
    elif method == "bhy":
        return _bhy_threshold(pvals, fdr=fdr, reshaping_function=reshaping_function)
    elif method == "ebh":
        return _ebh_threshold(pvals, fdr=fdr)
    else:
        raise ValueError("{} is not support FDR control method".format(method))


def cal_fdp_power(selected, non_zero_index, r_index=False):
    """Calculate power and False Discovery Proportion

    Parameters
    ----------
    selected: list index (in R format) of selected non-null variables
    non_zero_index: true index of non-null variables
    r_index : True if the index is taken from rpy2 inference

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

    if r_index:
        selected = selected - 1

    true_positive = np.intersect1d(selected, non_zero_index)
    false_positive = np.setdiff1d(selected, true_positive)

    fdp = len(false_positive) / max(1, len(selected))
    power = min(len(true_positive), n_positives) / n_positives

    return fdp, power


def _bhq_threshold(pvals, fdr=0.1):
    """Standard Benjamini-Hochberg for controlling False discovery rate"""
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
        return -1.0


def _ebh_threshold(evals, fdr=0.1):
    """e-BH procedure for FDR control (see Wang and Ramdas 2021)"""
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
        return np.infty


def _bhy_threshold(pvals, reshaping_function=None, fdr=0.1):
    """Benjamini-Hochberg-Yekutieli procedure for controlling FDR, with input
    shape function. Reference: Ramdas et al (2017)
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
            return -1.0


def _fixed_quantile_aggregation(pvals, gamma=0.5):
    """Quantile aggregation function based on Meinshausen et al (2008)

    Parameters
    ----------
    pvals : 2D ndarray (n_sampling_with_repetition, n_test)
        p-value (adjusted)

    gamma : float
        Percentile value used for aggregation.

    Returns
    -------
    1D ndarray (n_tests, )
        Vector of aggregated p-values
    """
    converted_score = (1 / gamma) * (np.percentile(pvals, q=100 * gamma, axis=0))

    return np.minimum(1, converted_score)


def _adaptive_quantile_aggregation(pvals, gamma_min=0.05):
    """adaptive version of the quantile aggregation method, Meinshausen et al.
    (2008)"""
    gammas = np.arange(gamma_min, 1.05, 0.05)
    list_Q = np.array([_fixed_quantile_aggregation(pvals, gamma) for gamma in gammas])

    return np.minimum(1, (1 - np.log(gamma_min)) * list_Q.min(0))


def _lambda_max(X, y, use_noise_estimate=True):
    """Calculation of lambda_max, the smallest value of regularization parameter in
    lasso program for non-zero coefficient
    """
    n_samples, _ = X.shape

    if not use_noise_estimate:
        return np.max(np.dot(X.T, y)) / n_samples

    norm_y = np.linalg.norm(y, ord=2)
    sigma_0 = (norm_y / np.sqrt(n_samples)) * 1e-3
    sig_star = max(sigma_0, norm_y / np.sqrt(n_samples))

    return np.max(np.abs(np.dot(X.T, y)) / (n_samples * sig_star))


def _check_vim_predict_method(method):
    """Check if the method is a valid method for variable importance measure
    prediction"""
    if method in ["predict", "predict_proba", "decision_function", "transform"]:
        return method
    else:
        raise ValueError(
            "The method {} is not a valid method for variable importance measure prediction".format(
                method
            )
        )
