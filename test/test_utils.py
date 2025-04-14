from hidimstat._utils.utils import (
    fdr_threshold,
    cal_fdp_power,
    quantile_aggregation,
    _alpha_max,
)
from hidimstat._utils.data_simulation import simu_data
from numpy.testing import assert_array_almost_equal
import numpy as np
import pytest

seed = 42


def test_fdr_threshold():
    """
    This function tests the application of the False Discovery Rate (FDR)
    control methods
    """
    p_values = np.linspace(1.0e-6, 1 - 1.0e-6, 100)
    p_values[:20] /= 10**6

    e_values = 1 / p_values

    identity = lambda i: i

    bhq_cutoff = fdr_threshold(p_values, fdr=0.1, method="bhq")
    bhy_cutoff = fdr_threshold(p_values, fdr=0.1, method="bhy")
    bhy_cutoff = fdr_threshold(
        p_values, fdr=0.1, method="bhy", reshaping_function=identity
    )
    ebh_cutoff = fdr_threshold(e_values, fdr=0.1, method="ebh")

    with pytest.raises(Exception):
        _ = fdr_threshold(e_values, fdr=0.1, method="test")

    # Test BHq
    assert len(p_values[p_values <= bhq_cutoff]) == 20

    # Test BHy
    assert len(p_values[p_values <= bhy_cutoff]) == 20

    # Test e-BH
    assert len(e_values[e_values >= ebh_cutoff]) == 20


def test_fdr_threshold_extreme_values():
    """test FDR computation for extreme numerical value of the p-values"""
    p_values = np.ones(100)
    e_values = 1 / p_values

    identity = lambda i: i

    bhq_cutoff = fdr_threshold(p_values, fdr=0.1, method="bhq")
    bhy_cutoff = fdr_threshold(p_values, fdr=0.1, method="bhy")
    bhy_cutoff = fdr_threshold(
        p_values, fdr=0.1, method="bhy", reshaping_function=identity
    )
    ebh_cutoff = fdr_threshold(e_values, fdr=0.1, method="ebh")

    # Test BHq
    assert bhq_cutoff == -1

    # Test BHy
    assert bhy_cutoff == -1

    # Test e-BH
    assert np.isinf(ebh_cutoff)


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

    # test empty selection
    fdp, power = cal_fdp_power(np.empty(0), non_zero_index)
    assert fdp == 0.0
    assert power == 0.0


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


def test_alpha_max():
    """Test alpha max function"""
    n = 500
    p = 100
    snr = 5
    X, y, beta_true, non_zero = simu_data(n, p, snr=snr, seed=0)
    max_alpha = _alpha_max(X, y)
    max_alpha_noise = _alpha_max(X, y, use_noise_estimate=True)
    # Assert alpha_max is positive
    assert max_alpha > 0
    assert max_alpha_noise > 0

    # Assert alpha_max with noise estimate is different (usually smaller)
    assert max_alpha_noise != max_alpha

    # Test with zero target vector
    y_zero = np.zeros_like(y)
    alpha_zero = _alpha_max(X, y_zero)
    assert alpha_zero == 0
