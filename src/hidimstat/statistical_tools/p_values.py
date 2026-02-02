import numpy as np
from scipy.stats import norm


def _replace_infinity(x, replace_val=None, method="times-two"):
    """
    Replace infinity values in array with finite values.

    This function replaces infinite values in an array with finite values based on the
    largest non-infinite value present in the array.

    Parameters
    ----------
    x : array-like
        Input array that may contain infinity values.
    replace_val : float, optional
        Custom replacement value for infinity. If provided and smaller than the
        calculated minimum replacement value, the minimum replacement value is used instead.
    method : {'times-two', 'plus-one'}, default='times-two'
        Method to calculate replacement value:
        - 'times-two': doubles the largest non-infinite absolute value
        - 'plus-one': adds 1 to the largest non-infinite absolute value

    Returns
    -------
    array-like
        Array with infinity values replaced by finite values.

    Notes
    -----
    The function preserves the sign of infinite values in the replacement.
    """
    largest_non_inf = np.max(np.abs(x)[np.abs(x) != np.inf])

    if method == "times-two":
        replace_val_min = largest_non_inf * 2
    elif method == "plus-one":
        replace_val_min = largest_non_inf + 1

    if (
        (replace_val is not None) and (replace_val < largest_non_inf)
    ) or replace_val is None:
        replace_val = replace_val_min

    x_new = np.copy(x)
    x_new[x_new == np.inf] = replace_val
    x_new[x_new == -np.inf] = -replace_val

    return x_new


def pval_corr_from_pval(one_sided_pval):
    """Computing one-sided p-values corrected for multiple testing
    from simple testing one-sided p-values.

    Parameters
    ----------
    one_sided_pval : ndarray, shape (n_features,)
        One-sided p-values.

    Returns
    -------
    one_sided_pval_corr : ndarray, shape (n_features,)
        Corrected one-sided p-values.
    """
    n_features = one_sided_pval.size

    one_sided_pval_corr = np.zeros(n_features) + 0.5

    ind = one_sided_pval < 0.5
    one_sided_pval_corr[ind] = np.minimum(
        one_sided_pval[ind] * n_features, 0.5
    )

    ind = one_sided_pval > 0.5
    one_sided_pval_corr[ind] = np.maximum(
        1 - (1 - one_sided_pval[ind]) * n_features, 0.5
    )

    return one_sided_pval_corr


def pval_from_scale(beta, scale, distribution="norm", eps=1e-14):
    """Computing one-sided p-values from the value of the parameter
    and its scale.

    Parameters
    ----------
    beta : ndarray, shape (n_features,)
        Value of the parameters.

    scale : ndarray, shape (n_features,)
        Value of the standard deviation of the parameters.

    distribution : str, optional (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    eps : float, optional
        Machine-precision regularization in the computation of the p-values.

    Returns
    -------
    pval : ndarray, shape (n_features,)
        p-value, with numerically accurate values for
        positive effects (ie., for p-value close to zero).

    pval_corr : ndarray, shape (n_features,)
        p-value corrected for multiple testing.

    one_minus_pval : ndarray, shape (n_features,)
        One minus the p-value, with numerically accurate values
        for negative effects (ie., for p-value close to one).

    one_minus_pval_corr : ndarray, shape (n_features,)
        One minus the p-value corrected for multiple testing.
    """
    n_features = beta.size

    index_no_nan = (scale != 0.0,)

    pval = np.zeros(n_features) + 0.5
    one_minus_pval = np.zeros(n_features) + 0.5

    if distribution == "norm":
        pval[index_no_nan] = norm.sf(
            beta[index_no_nan], scale=scale[index_no_nan]
        )
        one_minus_pval[index_no_nan] = norm.cdf(
            beta[index_no_nan], scale=scale[index_no_nan]
        )

    pval[pval > 1 - eps] = 1 - eps
    pval_corr = pval_corr_from_pval(pval)

    one_minus_pval[one_minus_pval > 1 - eps] = 1 - eps
    one_minus_pval_corr = pval_corr_from_pval(one_minus_pval)

    return pval, pval_corr, one_minus_pval, one_minus_pval_corr


def zscore_from_cb(cb_min, cb_max, confidence=0.95, distribution="norm"):
    """Computing z-scores from confidence intervals.

    Parameters
    ----------
    cb_min : ndarray, shape (n_features,)
        Value of the inferior confidence bound.

    cb_max : ndarray, shape (n_features,)
        Value of the superior confidence bound.

    confidence : float, optional (default=0.95)
        Confidence level used to compute the confidence intervals.
        Each value should be in the range [0, 1].

    distribution : str, optional (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    Returns
    -------
    zscore : ndarray, shape (n_features,)
        z-scores.
    """
    if distribution == "norm":
        quantile = norm.ppf(1 - (1 - confidence) / 2)

    beta_hat = (cb_min + cb_max) / 2

    zscore = beta_hat / (cb_max - cb_min) * 2 * quantile

    return zscore


def pval_from_cb(
    cb_min, cb_max, confidence=0.95, distribution="norm", eps=1e-14
):
    """Computing one-sided p-values from confidence intervals.

    Parameters
    ----------
    cb_min : ndarray, shape (n_features,)
        Value of the inferior confidence bound.

    cb_max : ndarray, shape (n_features,)
        Value of the superior confidence bound.

    confidence : float, optional (default=0.95)
        Confidence level used to compute the confidence intervals.
        Each value should be in the range [0, 1].

    distribution : str, optional (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    eps : float, optional
        Machine-precision regularization in the computation of the p-values.

    Returns
    -------
    pval : ndarray, shape (n_features,)
        p-value, with numerically accurate values for
        positive effects (ie., for p-value close to zero).

    pval_corr : ndarray, shape (n_features,)
        p-value corrected for multiple testing.

    one_minus_pval : ndarray, shape (n_features,)
        One minus the p-value, with numerically accurate values
        for negative effects (ie., for p-value close to one).

    one_minus_pval_corr : ndarray, shape (n_features,)
        One minus the p-value corrected for multiple testing.
    """
    zscore = zscore_from_cb(
        cb_min, cb_max, confidence=confidence, distribution=distribution
    )

    if distribution == "norm":
        pval = norm.sf(zscore)
        one_minus_pval = norm.cdf(zscore)

    pval[pval > 1 - eps] = 1 - eps
    pval_corr = pval_corr_from_pval(pval)

    one_minus_pval[one_minus_pval > 1 - eps] = 1 - eps
    one_minus_pval_corr = pval_corr_from_pval(one_minus_pval)

    return pval, pval_corr, one_minus_pval, one_minus_pval_corr


def two_sided_pval_from_zscore(zscore, distribution="norm"):
    """Computing two-sided p-values from z-scores.

    Parameters
    ----------
    zscore : ndarray, shape (n_features,)
        z-scores.

    distribution : str, optional (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    Returns
    -------
    two_sided_pval : ndarray, shape (n_features,)
        Two-sided p-values (testing the null).

    two_sided_pval_corr : ndarray, shape (n_features,)
        Two-sided p-values corrected for multiple testing.
    """
    n_features = zscore.size

    if distribution == "norm":
        two_sided_pval = 2 * norm.sf(np.abs(zscore))

    two_sided_pval_corr = np.minimum(1, two_sided_pval * n_features)

    return two_sided_pval, two_sided_pval_corr


def two_sided_pval_from_cb(
    cb_min, cb_max, confidence=0.95, distribution="norm"
):
    """Computing two-sided p-values from confidence intervals.

    Parameters
    ----------
    cb_min : ndarray, shape (n_features,)
        Value of the inferior confidence bound.

    cb_max : ndarray, shape (n_features,)
        Value of the superior confidence bound.

    confidence : float, optional (default=0.95)
        Confidence level used to compute the confidence intervals.
        Each value should be in the range [0, 1].

    distribution : str, optional (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    Returns
    -------
    two_sided_pval : ndarray, shape (n_features,)
        Two-sided p-values (testing the null).

    two_sided_pval_corr : ndarray, shape (n_features,)
        Two-sided p-values corrected for multiple testing.
    """
    zscore = zscore_from_cb(
        cb_min, cb_max, confidence=confidence, distribution=distribution
    )

    two_sided_pval, two_sided_pval_corr = two_sided_pval_from_zscore(
        zscore, distribution=distribution
    )

    return two_sided_pval, two_sided_pval_corr


def zscore_from_pval(pval, one_minus_pval=None, distribution="norm"):
    """Computing z-scores from one-sided p-values.

    Parameters
    ----------
    pval : ndarray, shape (n_features,)
        p-value, with numerically accurate values for
        positive effects (ie., for p-value close to zero).

    one_minus_pval : ndarray, shape (n_features,), optional (default=None)
        One minus the p-value, with numerically accurate values
        for negative effects (ie., for p-value close to one).

    distribution : str, optional (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    Returns
    -------
    zscore : ndarray, shape (n_features,)
        z-scores.
    """
    if distribution == "norm":
        zscore = norm.isf(pval)

        if one_minus_pval is not None:
            ind = pval > 0.5
            zscore[ind] = norm.ppf(one_minus_pval[ind])

    zscore = _replace_infinity(zscore, replace_val=40, method="plus-one")

    return zscore


def pval_from_two_sided_pval_and_sign(
    two_sided_pval, parameter_sign, eps=1e-14
):
    """Computing one-sided p-values from two-sided p-value and parameter sign.

    Parameters
    ----------
    two_sided_pval : ndarray, shape (n_features,)
        Two-sided p-values (testing the null).

    parameter_sign : ndarray, shape (n_features,)
        Estimated signs for the parameters.

    eps : float, optional
        Machine-precision regularization in the computation of the p-values.

    Returns
    -------
    pval : ndarray, shape (n_features,)
        p-value, with numerically accurate values for
        positive effects (ie., for p-value close to zero).

    pval_corr : ndarray, shape (n_features,)
        p-value corrected for multiple testing.

    one_minus_pval : ndarray, shape (n_features,)
        One minus the p-value, with numerically accurate values
        for negative effects (ie., for p-value close to one).

    one_minus_pval_corr : ndarray, shape (n_features,)
        One minus the p-value corrected for multiple testing.
    """
    n_features = two_sided_pval.size

    pval = 0.5 * np.ones(n_features)
    one_minus_pval = 0.5 * np.ones(n_features)

    pval[parameter_sign > 0] = two_sided_pval[parameter_sign > 0] / 2
    pval[parameter_sign < 0] = 1 - two_sided_pval[parameter_sign < 0] / 2

    one_minus_pval[parameter_sign > 0] = (
        1 - two_sided_pval[parameter_sign > 0] / 2
    )
    one_minus_pval[parameter_sign < 0] = two_sided_pval[parameter_sign < 0] / 2

    pval[pval > 1 - eps] = 1 - eps
    pval_corr = pval_corr_from_pval(pval)

    one_minus_pval[one_minus_pval > 1 - eps] = 1 - eps
    one_minus_pval_corr = pval_corr_from_pval(one_minus_pval)

    return pval, pval_corr, one_minus_pval, one_minus_pval_corr


def two_sided_pval_from_pval(pval, one_minus_pval=None, distribution="norm"):
    """Computing two-sided p-value from one-sided p-values.

    Parameters
    ----------
    pval : ndarray, shape (n_features,)
        p-value, with numerically accurate values for
        positive effects (ie., for p-value close to zero).

    one_minus_pval : ndarray, shape (n_features,), optional (default=None)
        One minus the p-value, with numerically accurate values
        for negative effects (ie., for p-value close to one).

    distribution : str, optional (default='norm')
        Type of distribution assumed for the underlying estimator.
        'norm' means normal and is the only value accepted at the moment.

    Returns
    -------
    two_sided_pval : ndarray, shape (n_features,)
        Two-sided p-values (testing the null).

    two_sided_pval_corr : ndarray, shape (n_features,)
        Two-sided p-values corrected for multiple testing.
    """
    zscore = zscore_from_pval(pval, one_minus_pval, distribution=distribution)

    two_sided_pval, two_sided_pval_corr = two_sided_pval_from_zscore(
        zscore, distribution=distribution
    )

    return two_sided_pval, two_sided_pval_corr
