from hidimstat.data_simulation import simu_data
from hidimstat.utils import fdr_threshold, cal_fdp_power, quantile_aggregation
from numpy.testing import assert_array_almost_equal
import numpy as np
import pytest

seed = 42


def test_fdr_threshold():
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

    null_p_values = np.linspace(1.0e-6, 1 - 1.0e-6, 100)
    null_e_values = 1 / null_p_values

    null_bhq_cutoff = fdr_threshold(null_p_values, fdr=0.1, method="bhq")
    null_bhy_cutoff = fdr_threshold(null_p_values, fdr=0.1, method="bhy")
    null_ebh_cutoff = fdr_threshold(null_e_values, fdr=0.1, method="ebh")

    # Test BHq on null data
    assert len(null_p_values[null_p_values <= null_bhq_cutoff]) <= 1

    # Test BHq on null data
    assert len(null_p_values[null_p_values <= null_bhy_cutoff]) <= 1

    # Test eBH on null data
    assert len(null_e_values[null_e_values >= null_ebh_cutoff]) <= 1


def test_cal_fdp_power():
    p_values = np.linspace(1.0e-6, 1 - 1.0e-6, 100)
    p_values[:20] /= 10**6

    selected = np.where(p_values < 1.0e-6)[0]
    # 2 False Positives and 3 False Negatives
    non_zero_index = np.concatenate([np.arange(18), [35, 36, 37]])

    fdp, power = cal_fdp_power(selected, non_zero_index)

    assert fdp == 2 / len(selected)
    assert power == 18 / len(non_zero_index)


def test_quantile_aggregation():
    col = np.arange(11)
    p_values = np.tile(col, (10, 1)).T / 100

    assert_array_almost_equal(0.1 * quantile_aggregation(p_values, 0.1), [0.01] * 10)
    assert_array_almost_equal(0.3 * quantile_aggregation(p_values, 0.3), [0.03] * 10)
    assert_array_almost_equal(0.5 * quantile_aggregation(p_values, 0.5), [0.05] * 10)
