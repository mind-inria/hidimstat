import numpy as np
import pytest

from hidimstat._utils.feature_selection import (
    fdr_selection,
    fwer_selection,
    generic_selection,
)


@pytest.fixture
def create_importances_pvalues(rng):
    """Create a BaseVariableImportance instance with test data for testing purposes.

    Parameters
    ----------
    rng: Python fixture
        Returns a Generator for random number generation.
    create_bvi: Python fixture
        Returns a new instance of hidimstat.BaseVariableImportance

    Returns
    -------
    BaseVariableImportance
        A BaseVariableImportance instance with test data.
    """
    n_features = 100
    importances = np.arange(n_features)
    rng.shuffle(importances)
    pvalues = np.flip(np.sort(rng.uniform(0, 1, n_features)))[importances]
    return importances, pvalues


def test_selection_generic_check_multiple_criteria(create_importances_pvalues):
    """Test selection multiple criteria error"""
    _importances, pvalues = create_importances_pvalues
    with pytest.raises(
        ValueError, match="Only support selection based on one criterion"
    ):
        generic_selection(pvalues, k_best=5, k_lowest=5)


def test_selection_generic_check_k_best(create_importances_pvalues):
    """Test selection k_best wrong"""
    _importances, pvalues = create_importances_pvalues
    with pytest.raises(
        ValueError, match="k_best needs to be strictly positive"
    ):
        generic_selection(pvalues, k_best=0)
    with pytest.warns(Warning, match="is greater than n_features="):
        generic_selection(pvalues, k_best=len(pvalues) + 1)


def test_selection_generic_check_k_lowest(create_importances_pvalues):
    """Test selection k_lowest wrong"""
    _importances, pvalues = create_importances_pvalues
    with pytest.raises(
        ValueError, match="k_lowest needs to be strictly positive"
    ):
        generic_selection(pvalues, k_lowest=0)
    with pytest.warns(Warning, match="is greater than n_features="):
        generic_selection(pvalues, k_lowest=len(pvalues) + 1)


def test_selection_generic_check_percentile(create_importances_pvalues):
    """Test selection percentile wrong"""
    _importances, pvalues = create_importances_pvalues
    with pytest.raises(
        ValueError,
        match=r"percentile must be between 0 and 100 \(exclusive\). Got -1.",
    ):
        generic_selection(pvalues, percentile=-1)
    with pytest.raises(
        ValueError,
        match=r"percentile must be between 0 and 100 \(exclusive\). Got 102.",
    ):
        generic_selection(pvalues, percentile=102)
    with pytest.raises(
        ValueError,
        match=r"percentile must be between 0 and 100 \(exclusive\). Got 0.",
    ):
        generic_selection(pvalues, percentile=0)
    with pytest.raises(
        ValueError,
        match=r"percentile must be between 0 and 100 \(exclusive\). Got 100",
    ):
        generic_selection(pvalues, percentile=100)


def test_fdr_selection_pvalues_check(create_importances_pvalues):
    """Test selection pvalues wrong"""
    importances, _ = create_importances_pvalues
    with pytest.raises(
        ValueError, match=r"'p_values' must be an numpy ndarray"
    ):
        fdr_selection(
            p_values=None,
            importances=importances,
            fdr=0.8,
        )
    with pytest.raises(
        ValueError, match=r"'p_values' must be an numpy ndarray"
    ):
        fdr_selection(
            p_values=list(range(3)),
            importances=importances,
            fdr=0.8,
        )


def test_fdr_selection_importances_check(create_importances_pvalues):
    """Test selection importances wrong"""
    _, pvalues = create_importances_pvalues
    with pytest.raises(
        ValueError, match=r"'importances' must be an numpy ndarray"
    ):
        fdr_selection(
            p_values=pvalues,
            importances=None,
            fdr=0.8,
        )
    with pytest.raises(
        ValueError, match=r"'importances' must be an numpy ndarray"
    ):
        fdr_selection(
            p_values=pvalues,
            importances=list(range(3)),
            fdr=0.8,
        )


def test_fdr_selection_pvalues_importances_shape_check():
    """Test selection pvalues and importances shape mismatch"""
    with pytest.raises(
        ValueError,
        match=r"Shape mismatch of 'p_values' and 'importances' on axis 0",
    ):
        fdr_selection(
            p_values=np.ones((10,)),
            importances=np.ones((15,)),
            fdr=0.8,
        )


def test_fdr_selection_fdr_check(create_importances_pvalues):
    """Test selection fdr wrong"""
    importances, pvalues = create_importances_pvalues
    with pytest.raises(
        ValueError,
        match=r"'fdr' must be a float strictly between 0 and 1, got -1.",
    ):
        fdr_selection(
            p_values=pvalues,
            importances=importances,
            fdr=-1,
        )
    with pytest.raises(
        ValueError,
        match=r"'fdr' must be a float strictly between 0 and 1, got 1.1.",
    ):
        fdr_selection(
            p_values=pvalues,
            importances=importances,
            fdr=1.1,
        )


def test_fdr_selection_fdr_control_check(create_importances_pvalues):
    """Test selection fdr_control wrong"""
    importances, pvalues = create_importances_pvalues
    with pytest.raises(
        ValueError,
        match=r"'fdr_control' must be one of \('bhq', 'bhy'\), got wrong.",
    ):
        fdr_selection(
            p_values=pvalues,
            importances=importances,
            fdr=0.8,
            fdr_control="wrong",
        )


def test_fwer_selection_pvalues_check(create_importances_pvalues):
    """Test selection pvalues wrong"""
    importances, _ = create_importances_pvalues
    with pytest.raises(
        ValueError, match=r"'p_values' must be an numpy ndarray"
    ):
        fwer_selection(
            p_values=None,
            importances=importances,
            fwer=0.8,
        )
    with pytest.raises(
        ValueError, match=r"'p_values' must be an numpy ndarray"
    ):
        fwer_selection(
            p_values=list(range(3)),
            importances=importances,
            fwer=0.8,
        )


def test_fwer_selection_importances_check(create_importances_pvalues):
    """Test selection importances wrong"""
    _, pvalues = create_importances_pvalues
    with pytest.raises(
        ValueError, match=r"'importances' must be an numpy ndarray"
    ):
        fwer_selection(
            p_values=pvalues,
            importances=None,
            fwer=0.8,
        )
    with pytest.raises(
        ValueError, match=r"'importances' must be an numpy ndarray"
    ):
        fwer_selection(
            p_values=pvalues,
            importances=list(range(3)),
            fwer=0.8,
        )


def test_fwer_selection_pvalues_importances_shape_check():
    """Test selection pvalues and importances shape mismatch"""
    with pytest.raises(
        ValueError,
        match=r"Shape mismatch of 'p_values' and 'importances' on axis 0",
    ):
        fwer_selection(
            p_values=np.ones((10,)),
            importances=np.ones((15,)),
            fwer=0.8,
        )


def test_fwer_selection_fwer_check(create_importances_pvalues):
    """Test selection fwer wrong"""
    importances, pvalues = create_importances_pvalues
    with pytest.raises(
        ValueError,
        match=r"'fwer' must be a float between 0 and 1, got -1.",
    ):
        fwer_selection(
            p_values=pvalues,
            importances=importances,
            fwer=-1,
        )
    with pytest.raises(
        ValueError,
        match=r"'fwer' must be a float between 0 and 1, got 1.1.",
    ):
        fwer_selection(
            p_values=pvalues,
            importances=importances,
            fwer=1.1,
        )


def test_fwer_selection_procedure_check(create_importances_pvalues):
    """Test selection procedure wrong"""
    importances, pvalues = create_importances_pvalues
    with pytest.raises(
        ValueError, match=r"Only 'bonferroni' procedure is supported"
    ):
        fwer_selection(
            p_values=pvalues,
            importances=importances,
            fwer=0.8,
            procedure="wrong",
        )


def test_fwer_selection_ntests_check(create_importances_pvalues):
    """Test selection n_tests wrong"""
    importances, pvalues = create_importances_pvalues
    with pytest.raises(
        ValueError, match=r"'n_tests' must be strictly positive."
    ):
        fwer_selection(
            p_values=pvalues, importances=importances, fwer=0.8, n_tests=0
        )
