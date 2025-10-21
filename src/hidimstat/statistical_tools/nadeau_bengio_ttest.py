from collections import namedtuple

import numpy as np
import scipy.special as special
from scipy.stats._stats_py import _var

NBTtestResult = namedtuple("NBTtestResult", ["statistic", "pvalue"])
NBTtestResult.__doc__ = "Class for Nadeau Bengio t-test"


def _chk_asarray(a, axis):
    """
    Based on https://github.com/scipy/scipy/blob/e407bc4d6ee71ec1a23fce9d95b58e28451e4d94/scipy/stats/_stats_py.py#L113
    """
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis

    if a.ndim == 0:
        a = np.atleast_1d(a)

    return a, outaxis


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
    One-sample t-test with variance corrected using Nadeau & Bengio.

    Simplification of https://github.com/scipy/scipy/blob/b1296b9b4393e251511fe8fdd3e58c22a1124899/scipy/stats/_stats_py.py#L6035-L6233
    Only support numpy backend and don't support masked data

    This is a modification of scipy.stats.ttest_1samp that applies the
    Nadeau & Bengio correction :footcite::`nadeau1999inference` to the variance
    estimate to account for dependence between repeated cross-validation estimates.

    Parameters
    ----------
    a : array_like
        Sample data. The axis specified by `axis` is the sample axis.
    popmean : scalar or array_like
        Expected value in null hypothesis. If array_like, must be
        broadcastable to the shape of the mean of `a` along `axis`.
    test_frac : float
        Fraction of the data used for testing (test set size / total
        samples). Used by the Nadeau & Bengio correction
        :footcite::`nadeau1999inference` when adjusting the sample variance.
    axis : int or None, optional
        Axis along which to compute the test. Default is 0. If None, the
        input array is flattened.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains NaNs. Default is
        'propagate'. Note: nan handling is performed by the
        `_axis_nan_policy` decorator; the parameter remains in the
        signature for compatibility.
    alternative : {'two-sided', 'greater', 'less'}, optional
        Defines the alternative hypothesis. Default is 'greater'.

    Returns
    -------
    TtestResult
        Named tuple with fields: statistic, pvalue, df, alternative,
        standard_error, estimate. The `statistic` and `pvalue` have the
        Nadeau & Bengio corrected standard error applied.

    Notes
    -----
    The variance is corrected using the factor proposed by Nadeau & Bengio :footcite::`nadeau1999inference`
    to account for dependence across repeated evaluations:
    corrected_var = var * (1/kr + test_frac),
    where kr is the number of repeated evaluations along `axis`.

    This function preserves the interface of scipy.stats.ttest_1samp while
    replacing the usual sample variance by the corrected variance.

    References
    ----------
    .. footbibliography::
    """
    a, axis = _chk_asarray(a, axis)

    n = a.shape[axis]
    df = n - 1

    if a.shape[axis] == 0:
        # This is really only needed for *testing* _axis_nan_policy decorator
        # It won't happen when the decorator is used.
        NaN = np.full((), np.nan, dtype=float)
        return NBTtestResult(NaN, NaN)

    mean = np.mean(a, axis=axis)
    try:
        popmean = np.asarray(popmean)
        popmean = np.squeeze(popmean, axis=axis) if popmean.ndim > 0 else popmean
    except ValueError as e:
        raise ValueError("`popmean.shape[axis]` must equal 1.") from e
    d = mean - popmean
    v = _var(a, axis=axis, ddof=1)
    ######### ADD correction of ttest
    denom = np.sqrt(v * (1 / n + test_frac))
    ################################

    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.divide(d, denom)
        t = t[()] if t.ndim == 0 else t

    prob = _get_pvalue(np.asarray(df, dtype=t.dtype), t, alternative)
    prob = prob[()] if prob.ndim == 0 else prob

    return NBTtestResult(t, prob)
