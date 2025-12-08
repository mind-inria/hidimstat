import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal

from hidimstat.statistical_tools.aggregation import (
    _adaptive_quantile_aggregation,
    _fixed_quantile_aggregation,
    quantile_aggregation,
)


def test_quantile_aggregation():
    """
    This function tests the application of the quantile aggregation method
    """
    nb_iter = 22
    nb_features = 10
    col = np.arange(nb_iter)
    p_values = np.tile(col, (nb_features, 1)).T / 100

    assert_array_almost_equal(
        0.1 * quantile_aggregation(p_values, 0.1), [col[-1] / 100 * 0.1] * nb_features
    )
    assert_array_almost_equal(
        0.3 * quantile_aggregation(p_values, 0.3), [col[-1] / 100 * 0.3] * nb_features
    )
    assert_array_almost_equal(
        0.5 * quantile_aggregation(p_values, 0.5), [col[-1] / 100 * 0.5] * nb_features
    )

    # with adaptation
    assert_array_almost_equal(
        quantile_aggregation(p_values, 0.1, adaptive=True) / (1 - np.log(0.1)),
        [nb_iter / 100] * nb_features,
    )
    assert_array_almost_equal(
        quantile_aggregation(p_values, 0.3, adaptive=True) / (1 - np.log(0.3)),
        [nb_iter / 100] * nb_features,
    )
    assert_array_almost_equal(
        quantile_aggregation(p_values, 0.5, adaptive=True) / (1 - np.log(0.5)),
        [nb_iter / 100] * nb_features,
    )

    # One p-value within the quantile aggregation method
    p_values = np.array([0.0])

    assert quantile_aggregation(p_values) == 0.0


def test_fixed_aggregate_quantiles():
    """Aggregated p-values is twice the median p-value. All p-values should
    be close to 0.04 and decreasing with respect to feature position."""

    n_iter, n_features = 20, 5
    list_pval = 1.0 / (np.arange(n_iter * n_features) + 1)
    list_pval = list_pval.reshape((n_iter, n_features))
    list_pval[15:, :] = 3e-3
    list_pval[:, 0] = 0.8

    pval = _fixed_quantile_aggregation(list_pval)
    expected = 0.04 * np.ones(n_features)
    expected[0] = 1.0

    assert_almost_equal(pval, expected, decimal=2)
    assert_equal(pval[-2] >= pval[-1], True)


def test_adaptive_quantiles():
    """Aggregated p-values from adaptive quantiles formula. All p-values should
    be close to 0.04 and decreasing with respect to feature position."""

    n_iter, n_features = 20, 5
    list_pval = 1.0 / (np.arange(n_iter * n_features) + 1)
    list_pval = list_pval.reshape((n_iter, n_features))
    list_pval[15:, :] = 3e-3

    pval = _adaptive_quantile_aggregation(list_pval, gamma_min=0.2)
    expected = 0.03 * np.ones(n_features)

    assert_almost_equal(pval, expected, decimal=2)
    assert_equal(pval[-2] >= pval[-1], True)
