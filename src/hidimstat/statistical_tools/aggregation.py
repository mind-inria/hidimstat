import numpy as np


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
        return _adaptive_quantile_aggregation(pvals, gamma)
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
    pvalue aggregate: 1D ndarray (n_tests, )
        Vector of aggregated p-values

    References
    ----------
    .. footbibliography::
    """
    assert gamma > 0 and gamma <= 1, "gamma should be between 0 and 1"
    # equation 2.2 of meinshausen2009pvalues
    converted_score = np.quantile(pvals, q=gamma, axis=0) / gamma
    return np.minimum(1, converted_score)


def _adaptive_quantile_aggregation(pvals, gamma_min=0.05):
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
