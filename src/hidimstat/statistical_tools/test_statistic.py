import numpy as np
from scipy._lib._array_api import array_namespace, xp_size
from scipy.stats._axis_nan_policy import _axis_nan_policy_factory
from scipy.stats._stats_py import (
    TtestResult,
    _chk_asarray,
    _get_nan,
    _get_pvalue,
    _SimpleStudentT,
    _var,
    is_marray,
    pack_TtestResult,
    unpack_TtestResult,
)


def _length_nonmasked(x, axis, keepdims=False, xp=None):
    """
    Copy from https://github.com/scipy/scipy/blob/2878daa7083375847e3a181553b146e843efcfad/scipy/_lib/_array_api.py#L584
    The copy was necessary for retrocompatibilities
    """
    xp = array_namespace(x) if xp is None else xp
    if is_marray(xp):
        if np.iterable(axis):
            message = "`axis` must be an integer or None for use with `MArray`."
            raise NotImplementedError(message)
        return xp.astype(xp.count(x, axis=axis, keepdims=keepdims), x.dtype)
    return (
        xp_size(x)
        if axis is None
        else
        # compact way to deal with axis tuples or ints
        int(np.prod(np.asarray(x.shape)[np.asarray(axis)]))
    )


def _var_nadeau_bengio(differences, test_frac, axis=0, ddof=0, mean=None, xp=None):
    """
    Adjust variance using the Nadeau & Bengio correction for repeated
    k-fold cross-validation (see :footcite:t:`nadeau1999inference`).

    Copy from https://github.com/scikit-learn/scikit-learn/blob/0c27a07f68e0eda7e1fcbce44a7615addec7f232/examples/model_selection/plot_grid_search_stats.py#L172
    This correction was proposed in :footcite:t:`nadeau1999inference`.

    Parameters
    ----------
    differences : array_like
        Array of differences (e.g. per-evaluation score differences). The axis
        given by `axis` corresponds to the repeated evaluations; kr =
        differences.shape[axis].
    test_frac : float
        Fraction of the data used for testing (test set size / total samples).
    axis : int, optional
        Axis along which to compute the variance. Default is 0.
    ddof : int, optional
        Degrees of freedom to use for variance estimation (passed to `_var`).
        Default is 0.
    mean : array_like or None, optional
        Precomputed mean along `axis` (passed to `_var`). If None, the mean is
        computed inside `_var`.
    xp : module or None, optional
        Array namespace (e.g. numpy or array-api compatible module). If None,
        the namespace is inferred from the input.

    Returns
    -------
    corrected_var : ndarray
        Variance corrected by the :footcite:t:`nadeau1999inference` factor:
        corrected_var = var * (1/kr + test_frac),
        where kr is the number of repeated evaluations along `axis` and `var`
        is the sample variance computed by `_var`.

    Notes
    -----
    This implements the variance correction proposed by
    :footcite:t:`nadeau1999inference` to account for the dependence between
    repeated cross-validation estimates.
    The function returns the corrected variance (not the standard deviation).

    References
    ----------
    .. footbibliography::
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = differences.shape[axis]

    var = _var(differences, ddof=ddof, axis=axis, mean=mean, xp=xp)
    corrected_var = var * (1 / kr + test_frac)
    return corrected_var


@_axis_nan_policy_factory(
    pack_TtestResult,
    default_axis=0,
    n_samples=2,
    result_to_tuple=unpack_TtestResult,
    n_outputs=6,
)
# nan_policy handled by `_axis_nan_policy`, but needs to be left
# in signature to preserve use as a positional argument
def ttest_1samp_corrected_NB(
    a,
    popmean,
    test_frac,
    axis=0,
    nan_policy="propagate",
    alternative="greater",
):
    """
    One-sample t-test with variance corrected using Nadeau & Bengio.

    Base on https://github.com/scipy/scipy/blob/b1296b9b4393e251511fe8fdd3e58c22a1124899/scipy/stats/_stats_py.py#L6035-L6233

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
    xp = array_namespace(a)
    a, axis = _chk_asarray(a, axis, xp=xp)

    n = _length_nonmasked(a, axis)
    df = n - 1

    if a.shape[axis] == 0:
        # This is really only needed for *testing* _axis_nan_policy decorator
        # It won't happen when the decorator is used.
        NaN = _get_nan(a)
        return TtestResult(
            NaN, NaN, df=NaN, alternative=NaN, standard_error=NaN, estimate=NaN
        )

    mean = xp.mean(a, axis=axis)
    try:
        popmean = xp.asarray(popmean)
        popmean = xp.squeeze(popmean, axis=axis) if popmean.ndim > 0 else popmean
    except ValueError as e:
        raise ValueError("`popmean.shape[axis]` must equal 1.") from e
    d = mean - popmean
    ##########################################################
    # Modification of the function
    # Correct the computation of variance
    v = _var_nadeau_bengio(a, test_frac, axis=axis, ddof=1)
    ##########################################################
    denom = xp.sqrt(v / n)

    with np.errstate(divide="ignore", invalid="ignore"):
        t = xp.divide(d, denom)
        t = t[()] if t.ndim == 0 else t

    dist = _SimpleStudentT(xp.asarray(df, dtype=t.dtype))
    prob = _get_pvalue(t, dist, alternative, xp=xp)
    prob = prob[()] if prob.ndim == 0 else prob

    # when nan_policy='omit', `df` can be different for different axis-slices
    df = xp.broadcast_to(xp.asarray(df), t.shape)
    df = df[()] if df.ndim == 0 else df
    # _axis_nan_policy decorator doesn't play well with strings
    alternative_num = {"less": -1, "two-sided": 0, "greater": 1}[alternative]
    return TtestResult(
        t,
        prob,
        df=df,
        alternative=alternative_num,
        standard_error=denom,
        estimate=mean,
        statistic_np=xp.asarray(t),
        xp=xp,
    )
