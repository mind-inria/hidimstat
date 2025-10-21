from collections import namedtuple

import numpy as np
import scipy.special as special
from scipy.stats._stats_py import _var

NBTtestResult = namedtuple("NBTtestResult", ["statistic", "pvalue"])
NBTtestResult.__doc__ = "Class for Nadeau Bengio t-test"


def _get_pvalue(df, statistic, alternative, symmetric=True):
    """
    Get p-value given the statistic, (continuous) distribution, and alternative
    Based on https://github.com/scipy/scipy/blob/e407bc4d6ee71ec1a23fce9d95b58e28451e4d94/scipy/stats/_stats_py.py#L1571
    """
    if alternative == "less":
        pvalue = special.stdtr(df, statistic)
    elif alternative == "greater":
        pvalue = special.stdtr(df, -statistic)
    elif alternative == "two-sided":
        pvalue = 2 * (
            special.stdtr(df, -statistic)
            if symmetric
            else np.minimum(special.stdtr(df, statistic), special.stdtr(df, -statistic))
        )
    else:
        message = "`alternative` must be 'less', 'greater', or 'two-sided'."
        raise ValueError(message)

    return pvalue


def nadeau_bengio_ttest(
    a,
    popmean,
    test_frac,
    axis=0,
    alternative="greater",
):
    """
    One-sample t-test with Nadeau & Bengio variance correction.

    Simplification of https://github.com/scipy/scipy/blob/b1296b9b4393e251511fe8fdd3e58c22a1124899/scipy/stats/_stats_py.py#L6035-L6233
    Remove all the check and the management of NaN and empty array.

    This is a modification of scipy.stats.ttest_1samp that applies the
    Nadeau & Bengio correction :footcite::`nadeau1999inference` to the variance
    estimate to account for dependence between repeated cross-validation estimates.

    Parameters
    ----------
    a : array_like
        Sample data. The axis specified by `axis` is the sample (observation)
        axis.
    popmean : scalar
        The population mean to test against.
    test_frac : float
        Fraction of the data used for testing (test set size / total
        samples). Used by the Nadeau & Bengio correction
        :footcite::`nadeau1999inference` when adjusting the sample variance.
    axis : int or None, optional
        Axis along which to compute the test. Default is 0. If None, the
        input array is flattened.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Present for API compatibility; this implementation does not perform
        special NaN handling (inputs should not contain NaNs).
    alternative : {'two-sided', 'greater', 'less'}, optional
        Defines the alternative hypothesis. Default is 'greater'.

    Returns
    -------
    NBTtestResult
        Named tuple with fields: statistic, pvalue. Both are computed using the
        Nadeau & Bengio corrected standard error.

    Notes
    -----
    The variance is corrected using the factor implemented here:
        corrected_var = var * (1 / n + test_frac)
    where n is the number of repeated evaluations along `axis`.

    This function does not support masked arrays and only accepts numpy arrays.

    References
    ----------
    .. footbibliography::
    """
    n = a.shape[axis]
    d = np.mean(a, axis=axis) - popmean
    v = _var(a, axis=axis, ddof=1)
    denom = np.sqrt(v * (1 / n + test_frac))
    t = np.divide(d, denom)
    prob = _get_pvalue(np.asarray(n - 1, dtype=t.dtype), t, alternative)
    return NBTtestResult(t, prob)
