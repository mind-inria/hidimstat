import numpy as np
from scipy.stats import t
from scipy.stats._stats_py import _var

from hidimstat.statistical_tools.utils import TestResult


def _get_pvalue(df, statistic, alternative, symmetric=True):
    """
    Get p-value given the statistic, (continuous) distribution, and alternative
    Based on https://github.com/scipy/scipy/blob/e407bc4d6ee71ec1a23fce9d95b58e28451e4d94/scipy/stats/_stats_py.py#L1571
    """
    if alternative == "less":
        pvalue = t.cdf(statistic, df)
    elif alternative == "greater":
        pvalue = t.sf(statistic, df)
    elif alternative == "two-sided":
        pvalue = 2 * (
            t.sf(statistic, df)
            if symmetric
            else np.minimum(t.cdf(statistic, df), t.sf(statistic, df))
        )
    else:
        message = "`alternative` must be 'less', 'greater', or 'two-sided'."
        raise ValueError(message)

    return pvalue


def nadeau_bengio_ttest(
    a,
    popmean,
    test_frac,
    alternative="greater",
):
    """
    One-sample t-test with Nadeau & Bengio variance correction.

    Simplification of https://github.com/scipy/scipy/blob/b1296b9b4393e251511fe8fdd3e58c22a1124899/scipy/stats/_stats_py.py#L6035-L6233
    Remove all the check and the management of NaN and empty array.

    This is a modification of scipy.stats.ttest_1samp that applies the
    :footcite:t:`nadeau1999inference` correction to the variance
    estimate to account for dependence between repeated cross-validation estimates.

    Parameters
    ----------
    a : array_like
        Sample data should be of shape (n_features, n_folds) or
        (n_features, n_permutations, n_folds) in which case it is averaged over
        permutations before computing the test statistic.
    popmean : scalar
        The population mean to test against.
    test_frac : float
        Fraction of the data used for testing (test set size / total
        samples). Used by the :footcite:t:`nadeau1999inference` correction
        when adjusting the sample variance.
    alternative : {'two-sided', 'greater', 'less'}, optional
        Defines the alternative hypothesis. Default is 'greater'.

    Returns
    -------
    TestResult
        Named tuple with fields: statistic, pvalue. Both are computed using the
        Nadeau & Bengio corrected standard error.

    Notes
    -----
    The variance is corrected using the factor implemented here:
    `corrected_var = var * (1 / n + test_frac)`
    where n is the number of repeated evaluations, i.e. the size of the last
    axis of `a`.

    This function does not support masked arrays and only accepts numpy arrays.

    References
    ----------
    .. footbibliography::
    """
    if a.ndim == 3:
        a_ = np.mean(a, axis=1)
    elif a.ndim == 2:
        a_ = a
    elif a.ndim == 1:
        a_ = a[np.newaxis, :]
    else:
        raise ValueError("Input array must be 1D, 2D, or 3D.")
    n = a_.shape[1]
    d = np.mean(a_, axis=1) - popmean
    v = _var(a_, axis=1, ddof=1)
    denom = np.sqrt(v * (1 / n + test_frac))
    t = np.divide(d, denom)
    prob = _get_pvalue(np.asarray(n - 1, dtype=t.dtype), t, alternative)
    return TestResult(t, prob)
