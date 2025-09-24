import numpy as np
from scipy import stats
from scipy.stats._result_classes import TtestResult


def corrected_std(differences, test_frac, axis=0):
    """
    Corrects standard deviation using Nadeau and Bengio's approach.
    Copy from https://github.com/jpaillard/permucate/blob/3af6ab91edbdf40b62a66fe9a31e45c168050c3a/permucate/utils.py#L198-L218

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = differences.shape[axis]
    corrected_var = np.var(differences, ddof=1, axis=axis) * (1 / kr + test_frac)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def _ttest_finish(df, t, alternative):
    """
    Common code between all 3 t-test functions.
    This is based on the following code:
    https://github.com/scipy/scipy/blob/5e4a5e3785f79dd4e8930eed883da89958860db2/scipy/stats/_stats_py.py#L6972-6994
    """
    # We use ``stdtr`` directly here as it handles the case when ``nan``
    # values are present in the data and masked arrays are passed
    # while ``t.cdf`` emits runtime warnings. This way ``_ttest_finish``
    # can be shared between the ``stats`` and ``mstats`` versions.

    # TODO need to check the validty of it
    if alternative == "less":
        pval = stats.t.sf(df, t)
    elif alternative == "greater":
        pval = stats.t.sf(df, -t)
    elif alternative == "two-sided":
        pval = stats.t.sf(df, -np.abs(t)) * 2
    else:
        raise ValueError("alternative must be " "'less', 'greater' or 'two-sided'")

    if t.ndim == 0:
        t = t[()]
    if pval.ndim == 0:
        pval = pval[()]

    return t, pval


def compute_corrected_ttest(
    differences,
    test_frac=0.1 / 0.9,
    axis=0,
    alternative="two-sided",
):
    """
    Computes right-tailed paired t-test with corrected variance.
    This is based on the following code:
    https://github.com/jpaillard/permucate/blob/3af6ab91edbdf40b62a66fe9a31e45c168050c3a/permucate/utils.py#L221-L252

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    df = differences.shape[axis] - 1

    mean = np.mean(differences, axis=axis)
    std = corrected_std(differences, test_frac, axis=axis)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.divide(mean, std)
    t, prob = _ttest_finish(df, t_stat, alternative)

    alternative_num = {"less": -1, "two-sided": 0, "greater": 1}[alternative]
    return TtestResult(
        statistic=t,
        pvalue=prob,
        df=df,
        alternative=alternative_num,
        standard_error=std,
        estimate=mean,
    )
