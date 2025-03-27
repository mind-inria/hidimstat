from hidimstat.utils import (
    fdr_threshold,
    cal_fdp_power,
    quantile_aggregation,
    _alpha_max,
    fixed_quantile_aggregation,
    adaptive_quantile_aggregation,
)
from hidimstat.data_simulation import simu_data
from numpy.testing import assert_array_almost_equal, assert_almost_equal, assert_equal
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

    pval = fixed_quantile_aggregation(list_pval)
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

    pval = adaptive_quantile_aggregation(list_pval, gamma_min=0.2)
    expected = 0.03 * np.ones(n_features)

    assert_almost_equal(pval, expected, decimal=2)
    assert_equal(pval[-2] >= pval[-1], True)


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
