import numpy as np
import pytest
from hidimstat.statistical_tools.multiple_testing import fdp_power, fdr_threshold
from hidimstat._utils.docstring import _aggregate_docstring


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


def test_fdp_power():
    """
    This function tests the computation of power and False Discovery Proportion
    (FDP)
    """
    p_values = np.linspace(1.0e-6, 1 - 1.0e-6, 100)
    p_values[:20] /= 10**6

    selected = np.where(p_values < 1.0e-6)[0]
    # 2 False Positives and 3 False Negatives
    non_zero_index = np.concatenate([np.arange(18), [35, 36, 37]])

    fdp, power = fdp_power(selected, non_zero_index)

    assert fdp == 2 / len(selected)
    assert power == 18 / len(non_zero_index)

    # test empty selection
    fdp, power = fdp_power(np.empty(0), non_zero_index)
    assert fdp == 0.0
    assert power == 0.0


def test_aggregate_docstring():
    """Test docstring grouping"""
    doc_quantile = "\n    Implements the quantile aggregation method for p-values based on :cite:meinshausen2009p.\n\n    The function aggregates multiple p-values into a single p-value while controlling\n\n    Parameters\n    ----------\n    pvals : ndarray of shape (n_sampling*2, n_test)\n        Matrix of p-values to aggregate. Each row represents a sampling instance.\n\n    Returns\n    -------\n    ndarray of shape (n_test,)\n        Vector of aggregated p-values, one for each hypothesis test.\n\n    References\n    ----------\n    .. footbibliography::\n\n    Notes\n    -----\n    The aggregated p-values are guaranteed to be valid p-values in [0,1].\n    "
    doc_fixed_quantile = "\n    Quantile aggregation function based on :cite:meinshausen2009p\n\n    Parameters\n    ----------\n    pvals : 2D ndarray (n_sampling*2, n_test)\n        p-valueur\n\n Returns\n    -------\n    1D ndarray (n_tests, )\n        Vector of aggregated p-values\n\n    References\n    ----------\n    .. footbibliography::\n    "
    doc_adaptive_quantile = "\n    Adaptive version of quantile aggregation method based on :cite:meinshausen2009p\n\n    Parameters\n    ----------\n    pvals : 2D ndarray (n_sampling*2, n_test)\n        p-value\n\n    Returns\n    -------\n    2D ndarray (n_tests, )\n        Vector of aggregated p-values\n\n    References\n    ----------\n    .. footbibliography::\n    "
    final_doc = _aggregate_docstring(
        [doc_quantile, None, doc_fixed_quantile, doc_adaptive_quantile],
        "Returns\n    -------\n    3D ndarray (n_tests, )\n        Vector of aggregated p-values\n",
    )
    assert (
        final_doc
        == "Implements the quantile aggregation method for p-values based on :cite:meinshausen2009p.\n\nThe function aggregates multiple p-values into a single p-value while controlling\nParameters\n----------\npvals : ndarray of shape (n_sampling*2, n_test)\nMatrix of p-values to aggregate. Each row represents a sampling instance.\npvals : 2D ndarray (n_sampling*2, n_test)\np-valueur\npvals : 2D ndarray (n_sampling*2, n_test)\np-value\nReturns\n-------\n3D ndarray (n_tests, )\nVector of aggregated p-values"
    )
