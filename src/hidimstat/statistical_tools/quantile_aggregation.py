import numpy as np

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
