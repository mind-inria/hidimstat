import numpy as np

from hidimstat.statistical_tools.utils import TestResult


def holdout_randomization_test(loss_diff, approx=False):
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
        \\mathbb{I}(\\Delta_k \\geq 0)\\right)

    In the cross-validated setting, one p-value is obtained per fold. They are
    combined either with a Bonferroni correction over the folds (the default),
    or by summing the loss differences over the folds before computing a single
    p-value (``approx=True``, Algorithm 4 of :footcite:t:`tansey2022holdout`).

    Parameters
    ----------
    loss_diff : ndarray of shape (n_features, n_permutations) or \
(n_features, n_permutations, n_folds)
        The loss differences between the two models for each feature and each
        permutation.
    approx : bool, default=False
        Whether to use the approximate version of the holdout randomization
        test (Algorithm 4). Only used when `loss_diff` is 3D.

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
    n_permutations = loss_diff.shape[1]
    p_values = (1 + np.sum(loss_diff >= 0, axis=1)) / (n_permutations + 1)
    if loss_diff.ndim == 2:
        return TestResult(None, p_values)
    elif loss_diff.ndim == 3:
        n_folds = loss_diff.shape[2]
        if approx:
            approx_pvalues = (
                1 + np.sum(np.sum(loss_diff, axis=2) >= 0, axis=1)
            ) / (n_permutations + 1)
            return TestResult(None, approx_pvalues)
        else:
            # Bonferroni over the folds, clipped so that the result stays a
            # valid p-value.
            corrected_p_values = np.minimum(
                1.0, n_folds * np.min(p_values, axis=1)
            )
            return TestResult(None, corrected_p_values)
    else:
        raise ValueError(
            "loss_diff must be a 2D or 3D array, but got an array with shape "
            f"{loss_diff.shape}."
        )
