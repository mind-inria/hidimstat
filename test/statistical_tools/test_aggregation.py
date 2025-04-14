import numpy as np
from numpy.testing import assert_array_almost_equal
from hidimstat.statistical_tools.aggregation import quantile_aggregation


def test_quantile_aggregation():
    """
    This function tests the application of the quantile aggregation method
    """
    col = np.arange(11)
    p_values = np.tile(col, (10, 1)).T / 100

    assert_array_almost_equal(0.1 * quantile_aggregation(p_values, 0.1), [0.01] * 10)
    assert_array_almost_equal(0.3 * quantile_aggregation(p_values, 0.3), [0.03] * 10)
    assert_array_almost_equal(0.5 * quantile_aggregation(p_values, 0.5), [0.05] * 10)

    # with adaptation
    assert_array_almost_equal(
        quantile_aggregation(p_values, 0.1, adaptive=True) / (1 - np.log(0.1)),
        [0.1] * 10,
    )
    assert_array_almost_equal(
        quantile_aggregation(p_values, 0.3, adaptive=True) / (1 - np.log(0.3)),
        [0.1] * 10,
    )
    assert_array_almost_equal(
        quantile_aggregation(p_values, 0.5, adaptive=True) / (1 - np.log(0.5)),
        [0.1] * 10,
    )

    # One p-value within the quantile aggregation method
    p_values = np.array([0.0])

    assert quantile_aggregation(p_values) == 0.0
