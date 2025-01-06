from hidimstat.utils import fdr_threshold, cal_fdp_power, quantile_aggregation
from numpy.testing import assert_array_almost_equal
import numpy as np
import pytest

seed = 42


def test_quantile_aggregation():
    """
    This function tests the application of the quantile aggregation method
    """
    col = np.arange(11)
    p_values = np.tile(col, (10, 1)).T / 100
    gamma_min = 0.05

    assert_array_almost_equal(0.1 * quantile_aggregation(p_values, 0.1), [0.01] * 10)
    assert_array_almost_equal(
        0.1
        * quantile_aggregation(p_values, 0.1, adaptive=True, gamma_min=gamma_min)
        / (1 - np.log(gamma_min)),
        [0.01] * 10,
    )
    assert_array_almost_equal(0.3 * quantile_aggregation(p_values, 0.3), [0.03] * 10)
    assert_array_almost_equal(
        0.3
        * quantile_aggregation(p_values, 0.3, adaptive=True, gamma_min=gamma_min)
        / (1 - np.log(gamma_min)),
        [0.03] * 10,
    )
    assert_array_almost_equal(0.5 * quantile_aggregation(p_values, 0.5), [0.05] * 10)
    assert_array_almost_equal(
        0.5
        * quantile_aggregation(p_values, 0.5, adaptive=True, gamma_min=gamma_min)
        / (1 - np.log(gamma_min)),
        [0.05] * 10,
    )

    # One p-value within the quantile aggregation method
    p_values = np.array([0.0])

    assert quantile_aggregation(p_values) == 0.0
    assert quantile_aggregation(p_values, adaptive=True) == 0.0


def test_fdr_threshold():
    """
    This function tests the application of the False Discovery Rate (FDR)
    control methods
    """
    p_values = np.linspace(1.0e-6, 1 - 1.0e-6, 100)
    p_values[:20] /= 10**6

    e_values = 1 / p_values

    bhq_cutoff = fdr_threshold(p_values, fdr=0.1, method="bhq")
    bhy_cutoff = fdr_threshold(p_values, fdr=0.1, method="bhy")
    ebh_cutoff = fdr_threshold(e_values, fdr=0.1, method="ebh")

    with pytest.raises(Exception):
        _ = fdr_threshold(e_values, fdr=0.1, method="test")

    # Test BHq
    assert len(p_values[p_values <= bhq_cutoff]) == 20

    # Test BHy
    assert len(p_values[p_values <= bhy_cutoff]) == 20

    # Test e-BH
    assert len(e_values[e_values >= ebh_cutoff]) == 20


def test_cal_fdp_power():
    """
    This function tests the computation of power and False Discovery Proportion
    (FDP)
    """
    p_values = np.linspace(1.0e-6, 1 - 1.0e-6, 100)
    p_values[:20] /= 10**6

    selected = np.where(p_values < 1.0e-6)[0]
    # 2 False Positives and 3 False Negatives
    non_zero_index = np.concatenate([np.arange(18), [35, 36, 37]])

    fdp, power = cal_fdp_power(selected, non_zero_index)

    assert fdp == 2 / len(selected)
    assert power == 18 / len(non_zero_index)

    # test for zero selection
    fdp, power = cal_fdp_power(np.array([]), non_zero_index)

    assert fdp == 0
    assert power == 0

    # test for shift in index
    fdp, power = cal_fdp_power(selected + 1, non_zero_index, r_index=True)

    assert fdp == 2 / len(selected)
    assert power == 18 / len(non_zero_index)
