import numpy as np

from hidimstat.statistical_tools.aggregation import quantile_aggregation
from hidimstat.statistical_tools.utils import TestResult


def holdout_randomization_test(loss_diff, gamma=0.5, adaptive=False):
    """
    Compute the p-values of the holdout randomization test (HRT).

    The procedure follows :footcite:t:`tansey2022holdout`.
    For a feature :math:`j` (resp. a feature group), `n_permutations` samples
    are drawn from the conditional distribution :math:`p(X_j \\mid X_{-j})`, and
    for each draw, the loss difference difference between the empirical risk
    obtained on the sampled data and the empirical risk obtained on the original
    data is computed, :math:`\\Delta_k = t_k - t_0`, with :math:`t_0` being the
    empirical risk obtained on the original data and :math:`t_k` being the
    empirical risk obtained on the sampled data. Then the p-value is computed
    using the following formula:

    .. math::
        p_j = \\frac{1}{K+1} \\left(1 + \\sum_{k=1}^{K}
        \\mathbb{I}(\\Delta_k \\leq 0)\\right)

    An important feature makes the resampled risks larger than the original
    one, so :math:`\\Delta_k > 0` for most draws and the p-value is small.

    In the cross-validated setting, the folds have to be combined. By default,
    one p-value is computed per fold and they are combined with a Bonferroni
    correction, which is valid but conservative. With ``approx=True``
    (Algorithm 4 of :footcite:t:`tansey2022holdout`), the loss differences are
    instead summed over the folds and a single p-value is computed from the
    resulting draws, as in the single-split case. Pooling the folds this way
    accumulates the evidence of all of them and is therefore less
    conservative, but it is only approximate: the k-th draw of a fold is
    arbitrarily paired with the k-th draw of the others.

    Parameters
    ----------
    loss_diff : ndarray
        The loss differences between the two models for each feature and each
        permutation. Should be of shape (n_features, n_permutations) or
        (n_features, n_permutations, n_folds), or (n_permutations,) for a
        single feature.
    gamma : float, default=0.5
        Quantile level for aggregation. Must be in range (0,1].
    adaptive : bool, default=False
        If True, uses adaptive quantile aggregation which optimizes over
        multiple gamma values. If False, uses fixed quantile aggregation with
        the provided gamma value.

    Returns
    -------
    TestResult
        Named tuple with fields: statistic, pvalue. The holdout randomization
        test does not define a test statistic, so `statistic` is always None
        and `pvalue` is an array of shape (n_features,).

    References
    ----------
    .. footbibliography::
    """
    if loss_diff.ndim == 1:
        loss_diff_ = loss_diff[np.newaxis, :]
    elif loss_diff.ndim in (2, 3):
        loss_diff_ = loss_diff
    else:
        raise ValueError(
            "loss_diff must be 1D, 2D, or 3D, but got an array with shape "
            f"{loss_diff.shape}."
        )
    n_permutations = loss_diff_.shape[1]
    p_values = (1 + np.sum(loss_diff_ <= 0, axis=1)) / (n_permutations + 1)
    if loss_diff_.ndim == 2:
        return TestResult(None, p_values)

    corrected_p_values = quantile_aggregation(
        p_values, gamma=gamma, adaptive=adaptive
    )
    return TestResult(None, corrected_p_values)
