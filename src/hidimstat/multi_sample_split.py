import numpy as np


def aggregate_medians(list_one_sided_pval):
    """Aggregation of survival function values taking twice the median

    For me details, see:footcite:t:`meinshausen2009pvalues`.

    Parameters
    ----------
    list_one_sided_pval : ndarray, shape (n_iter, n_features)
        List of one-sided p-values.

    Returns
    -------
    one_sided_pval : ndarray, shape (n_features,)
        Aggregated one-sided p-values.

    References
    ----------
    .. footbibliography::
    """

    n_iter, n_features = list_one_sided_pval.shape

    one_sided_pval = np.median(list_one_sided_pval, axis=0)
    one_sided_pval[one_sided_pval > 0.5] = np.maximum(
        0.5, 1 - (1 - one_sided_pval[one_sided_pval > 0.5]) * 2
    )
    one_sided_pval[one_sided_pval < 0.5] = np.minimum(
        0.5, one_sided_pval[one_sided_pval < 0.5] * 2
    )

    return one_sided_pval


def aggregate_quantiles(list_one_sided_pval, gamma_min=0.2):
    """Aggregation of survival function values by adaptive quantile procedure
    :footcite:t:`meinshausen2009pvalues`

    Parameters
    ----------
    list_one_sided_pval : ndarray, shape (n_iter, n_features)
        List of one-sided p-values.

    gamma_min : float, optional (default=0.2)
        Lowest gamma-quantile being considered to compute the adaptive
        quantile aggregation formula.

    Returns
    -------
    one_sided_pval : ndarray, shape (n_features,)
        Aggregated one-sided p-values.

    References
    ----------
    .. footbibliography::
    """

    n_iter, n_features = list_one_sided_pval.shape
    one_sided_pval = 0.5 * np.ones(n_features)

    m = n_iter + 1
    k = np.maximum(1, int(np.floor(gamma_min * n_iter)))
    r = 1 - np.log(gamma_min)
    seq = range(k, n_iter)

    ordered_pval = np.sort(list_one_sided_pval, axis=0)
    rev_ordered_pval = ordered_pval[::-1]

    for i in np.arange(n_features):

        adjusted_ordered_pval = min([ordered_pval[j, i] * m / (j + 1) for j in seq])
        adjusted_ordered_pval = min(0.5, adjusted_ordered_pval)

        adjusted_rev_ordered_pval = max(
            [1 - (1 - rev_ordered_pval[j, i]) * m / (j + 1) for j in seq]
        )
        adjusted_rev_ordered_pval = max(0.5, adjusted_rev_ordered_pval)

        if (1 - adjusted_rev_ordered_pval) < adjusted_ordered_pval:

            one_sided_pval[i] = np.maximum(0.5, 1 - (1 - adjusted_rev_ordered_pval) * r)

        else:

            one_sided_pval[i] = np.minimum(0.5, adjusted_ordered_pval * r)

    return one_sided_pval
