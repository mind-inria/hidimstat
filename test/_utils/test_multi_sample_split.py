"""
Test the multi_sample_split module
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from hidimstat.statistical_tools.multi_sample_split import aggregate_medians, aggregate_quantiles


def test_aggregate_medians():
    """Aggregated p-values is twice the median p-value. All p-values should
    be close to 0.04 and decreasing with respect to feature position."""

    n_iter, n_features = 20, 5
    list_pval = 1.0 / (np.arange(n_iter * n_features) + 1)
    list_pval = list_pval.reshape((n_iter, n_features))
    list_pval[15:, :] = 3e-3

    pval = aggregate_medians(list_pval)
    expected = 0.04 * np.ones(n_features)

    assert_almost_equal(pval, expected, decimal=2)
    assert_equal(pval[-2] >= pval[-1], True)


def test_aggregate_quantiles():
    """Aggregated p-values from adaptive quantiles formula. All p-values should
    be close to 0.04 and decreasing with respect to feature position."""

    n_iter, n_features = 20, 5
    list_pval = 1.0 / (np.arange(n_iter * n_features) + 1)
    list_pval = list_pval.reshape((n_iter, n_features))
    list_pval[15:, :] = 3e-3

    pval = aggregate_quantiles(list_pval)
    expected = 0.03 * np.ones(n_features)

    assert_almost_equal(pval, expected, decimal=2)
    assert_equal(pval[-2] >= pval[-1], True)
