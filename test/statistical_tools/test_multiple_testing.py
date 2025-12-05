import numpy as np
import pytest

from hidimstat._utils.docstring import _aggregate_docstring
from hidimstat.statistical_tools.multiple_testing import fdp_power, fdr_threshold


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
    assert ebh_cutoff == 1


def test_fdp_power():
    """
    This function tests the computation of power and False Discovery Proportion
    (FDP)
    """
    p_values = np.linspace(1.0e-6, 1 - 1.0e-6, 100)
    p_values[:20] /= 10**6

    selected = np.zeros(100, dtype=int)
    selected[p_values < 1.0e-6] = 1
    # 2 False Positives and 3 False Negatives
    non_zero_mask = np.zeros(100, dtype=int)
    non_zero_mask[np.concatenate([np.arange(18), [35, 36, 37]])] = 1

    fdp, power = fdp_power(selected, non_zero_mask)

    assert fdp == 2 / np.sum(selected)
    assert power == 18 / np.sum(non_zero_mask)

    # test empty selection
    fdp, power = fdp_power(np.zeros(100, dtype=int), non_zero_mask)
    assert fdp == 0.0
    assert power == 0.0


def test_aggregate_docstring():
    """Test docstring grouping"""
    doc_minimum_1 = """
    Short Summary
    
    Parameters
    ----------
    param_1: ndarray of shape (n_sampling,)
        short description
    
    param_2: float, default=2.0
        short description 2
    
    Returns
    -------
    ndarray of shape (n_sampling,)
        short description for return
    
    References
    ----------
    .. footbibliography::
    
    Notes
    -----
    
    Complementary information
    """
    doc_minimum_2 = """
    Name object
    
    Description
    
    Parameters
    ----------
    param2_1: float, default=2.0
        description param2_1
    
    param2_2: ndarray of shape (n_sampling,)
        description param2_2
    
    Returns
    -------
    time_example (float)
        description of return
    """
    doc_minimum_3 = """
    Short Description
    
    Parameters
    ----------
    param_first: int, default=10
        integer interpretation
    
    param_second: float, default=2.0
        float interpretation
    
    Returns
    -------
    None
        
    
    References
    ----------
    .. footbibliography::
    
    """
    final_doc = _aggregate_docstring(
        [doc_minimum_1, None, doc_minimum_2, doc_minimum_3],
        """
        Returns
        -------
        3D ndarray (n_tests, )
        Vector of aggregated p-values
        """,
    )
    assert (
        final_doc
        == """Short Summary
Parameters
----------
param_1: ndarray of shape (n_sampling,)
short description

param_2: float, default=2.0
short description 2
param2_1: float, default=2.0
description param2_1

param2_2: ndarray of shape (n_sampling,)
description param2_2
param_first: int, default=10
integer interpretation

param_second: float, default=2.0
float interpretation

Returns
-------
3D ndarray (n_tests, )
Vector of aggregated p-values"""
    )
