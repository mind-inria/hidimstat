import numpy as np
from scipy._lib._array_api import (
    array_namespace,
)
from scipy.stats._axis_nan_policy import _axis_nan_policy_factory
from scipy.stats._stats_py import (
    TtestResult,
    _chk_asarray,
    _get_nan,
    _get_pvalue,
    _length_nonmasked,
    _SimpleStudentT,
    _var,
    pack_TtestResult,
    unpack_TtestResult,
)


def _var_nadeau_bengio(differences, test_frac, axis=0, ddof=0, mean=None, xp=None):
    """
    Corrects standard deviation using Nadeau and Bengio's approach.

    Copy from https://github.com/scikit-learn/scikit-learn/blob/0c27a07f68e0eda7e1fcbce44a7615addec7f232/examples/model_selection/plot_grid_search_stats.py#L172
    This correction was proposed in :footcite:t:`nadeau1999inference`

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.

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
    Modification of https://github.com/scipy/scipy/blob/b1296b9b4393e251511fe8fdd3e58c22a1124899/scipy/stats/_stats_py.py#L6035-L6233
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
    # Correct the computation of variance
    v = _var_nadeau_bengio(a, test_frac, axis=axis, ddof=1)
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
