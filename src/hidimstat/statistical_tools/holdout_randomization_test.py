import numpy as np

from hidimstat.statistical_tools.utils import TtestResult


def holdout_randomization_test(loss_diff):
    """
    Compute the p-value of the holdout randomization, following the procedure
    described in :footcite:t:`tansey2022holdout`.

    For a feature :math:`j` (resp. a feature group), `n_permutations` samples
    are drawn from the conditional distribution :math:`p(X_j \\mid X_{-j})`, and
    for each draw, the loss difference difference between the empirical risk
    obtained on the sampled data and the empirical risk obtained on the original
    data is computed, :math:`\\Delta_k = t_k - t_0`, with :math:`t_0` being the
    empirical risk obtained on the original data and :math:`t_k` being the
    empirical risk obtained on the sampled data. Then the p-value is computed
    using the following formula:

    .. math::
        p_j = \\frac{1}{K+1} \\sum_{k=1}^{K} \\mathbb{I}(\\Delta_k \\geq 0)


    Parameters
    ----------
    loss_diff : array-like of shape (n_features, n_permutations)
        The loss differences between the two models for each feature and each
        permutation.

    Returns
    -------
    p_value : float
        The p-value of the holdout randomization test.
    """
    n_permutations = loss_diff.shape[1]

    p_values = (1 + np.sum(loss_diff >= 0, axis=1)) / (n_permutations + 1)
    return TtestResult(None, p_values)
