import numpy as np
from copy import deepcopy


########################## quantile aggregation method ##########################
def quantile_aggregation(pvals, gamma=0.05, n_grid=20, adaptive=False):
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
        The quantile level (between 0 and 1) used for aggregation.
        For non-adaptive aggregation, a single gamma value is used.
        For adaptive aggregation, this is the starting point for the grid search
        over gamma values.
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
    1D ndarray (n_tests, )
        Vector of aggregated p-values

    References
    ----------
    .. footbibliography::
    """
    assert gamma > 0 and gamma <= 1, "gamma should be between O and 1"
    # equation 2.2 of meinshausen2009pvalues
    converted_score = (1 / gamma) * (np.percentile(pvals, q=100 * gamma, axis=0))
    return np.minimum(1, converted_score)


def _adaptive_quantile_aggregation(pvals, gamma_min=0.05, n_grid=20):
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


def detection_section(lines):
    """
    Detect sections in a numpy-style docstring by identifying section headers and their underlines.

    Parameters
    ----------
    lines : list of str
        Lines of the docstring to parse.

    Returns
    -------
    list of list of str
        List of sections, where each section is a list of lines belonging to that section.
        The first section is the summary, followed by other sections like Parameters, Returns, etc.
    """
    sections = []
    index_line = 1
    begin_section = index_line
    while len(lines) > index_line:
        if "-------" in lines[index_line]:
            sections.append(lines[begin_section : index_line - 2])
            begin_section = index_line - 1
        index_line += 1
    sections.append(lines[begin_section : len(lines)])
    return sections


def parse_docstring(docstring):
    """
    Parse a numpy-style docstring into its component sections.

    Parameters
    ----------
    docstring : str
        The docstring to parse, following numpy docstring format.

    Returns
    -------
    dict
        Dictionary containing docstring sections with keys like 'short' (summary),
        'Parameters', 'Returns', etc. Values are the text content of each section.
    """
    lines = docstring.split("\n")
    section_texts = detection_section(lines)
    sections = {"short": section_texts[0]}
    for section_text in section_texts:
        if len(section_text) <= 1 or "---" not in section_text[1]:
            sections["short"] = section_text
        else:
            sections["".join(section_text[0].split())] = section_text
    return sections


def reindent(string):
    """
    Reindent a string by stripping whitespace and normalizing line breaks.

    Parameters
    ----------
    string : list of str
        The string content to reindent.

    Returns
    -------
    str
        Reindented string with normalized line breaks and indentation.
    """
    new_string = deepcopy(string)
    for i in range(len(new_string)):
        new_string[i] = "\n" + new_string[i]
    new_string = "".join(new_string)
    return "\n".join(l.strip() for l in new_string.strip().split("\n"))


def aggregate_docstring(list_docstring):
    """
    Combine multiple docstrings into a single docstring.

    This function takes a list of docstrings, parses each one, and combines them into
    a single coherent docstring. It keeps the summary from the first docstring,
    combines all parameter sections, and uses the return section from the last docstring.

    Parameters
    ----------
    list_docstring : list
        List of docstrings to be combined. Each docstring should follow
        numpy docstring format.

    Returns
    -------
    doctring: str
        A combined docstring containing:
        - Summary from first docstring
        - Combined parameters from all docstrings
        - Returns section from last docstring
        The returned docstring is properly reindented.
    """
    list_line = []
    for index, docstring in enumerate(list_docstring):
        if docstring is not None:
            list_line.append(parse_docstring(docstring=docstring))

    # add summary
    final_docstring = deepcopy(list_line[0]["short"])
    # add parameter
    final_docstring += list_line[0]["Parameters"]
    for i in range(1, len(list_line)):
        # add paraemter after remove the title section
        final_docstring += list_line[i]["Parameters"][2:]
    # the last return
    final_docstring += list_line[-1]["Returns"]
    return reindent(final_docstring)
