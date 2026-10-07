import numpy as np
import pytest

from hidimstat._utils.selection import (
    _selection_fdr,
    _selection_fwer,
    _selection_generic,
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


class TestSelectionGenericParameterCheck:
    """Test class for '_selection_generic' parameter checks"""

    def test_selection_multiple_criteria(self, create_importances_pvalues):
        """Test selection multiple criteria error"""
        _importances, pvalues = create_importances_pvalues
        with pytest.raises(
            ValueError, match="Only support selection based on one criteria"
        ):
            _selection_generic(pvalues, k_best=5, k_lowest=5)

    def test_selection_k_best(self, create_importances_pvalues):
        """Test selection k_best wrong"""
        _importances, pvalues = create_importances_pvalues
        with pytest.raises(
            ValueError, match="k_best needs to be strictly positive"
        ):
            _selection_generic(pvalues, k_best=0)
        with pytest.warns(Warning, match="is greater than n_features="):
            _selection_generic(pvalues, k_best=len(pvalues) + 1)

    def test_selection_k_lowest(self, create_importances_pvalues):
        """Test selection k_lowest wrong"""
        _importances, pvalues = create_importances_pvalues
        with pytest.raises(
            ValueError, match="k_lowest needs to be strictly positive"
        ):
            _selection_generic(pvalues, k_lowest=0)
        with pytest.warns(Warning, match="is greater than n_features="):
            _selection_generic(pvalues, k_lowest=len(pvalues) + 1)

    def test_selection_percentile(self, create_importances_pvalues):
        """Test selection percentile wrong"""
        _importances, pvalues = create_importances_pvalues
        with pytest.raises(
            ValueError,
            match=r"percentile must be between 0 and 100 \(exclusive\). Got -1.",
        ):
            _selection_generic(pvalues, percentile=-1)
        with pytest.raises(
            ValueError,
            match=r"percentile must be between 0 and 100 \(exclusive\). Got 102.",
        ):
            _selection_generic(pvalues, percentile=102)
        with pytest.raises(
            ValueError,
            match=r"percentile must be between 0 and 100 \(exclusive\). Got 0.",
        ):
            _selection_generic(pvalues, percentile=0)
        with pytest.raises(
            ValueError,
            match=r"percentile must be between 0 and 100 \(exclusive\). Got 100",
        ):
            _selection_generic(pvalues, percentile=100)


class TestSelectionFDRParameterChecks:
    """Test class for 'selection_fdr' parameter checks"""

    def test_selection_fdr_pvalues_check(self, create_importances_pvalues):
        """Test selection pvalues wrong"""
        importances, _ = create_importances_pvalues
        with pytest.raises(
            ValueError, match=r"'p_values' must be an numpy ndarray"
        ):
            _selection_fdr(
                p_values=None,
                importances=importances,
                fdr=0.8,
            )
        with pytest.raises(
            ValueError, match=r"'p_values' must be an numpy ndarray"
        ):
            _selection_fdr(
                p_values=list(range(3)),
                importances=importances,
                fdr=0.8,
            )

    def test_selection_fdr_importances_check(self, create_importances_pvalues):
        """Test selection importances wrong"""
        _, pvalues = create_importances_pvalues
        with pytest.raises(
            ValueError, match=r"'importances' must be an numpy ndarray"
        ):
            _selection_fdr(
                p_values=pvalues,
                importances=None,
                fdr=0.8,
            )
        with pytest.raises(
            ValueError, match=r"'importances' must be an numpy ndarray"
        ):
            _selection_fdr(
                p_values=pvalues,
                importances=list(range(3)),
                fdr=0.8,
            )

    def test_selection_fdr_pvalues_importances_shape_check(self):
        """Test selection pvalues and importances shape mismatch"""
        with pytest.raises(
            ValueError,
            match=r"Shape mismatch of 'p_values' and 'importances' on axis 0",
        ):
            _selection_fdr(
                p_values=np.ones((10,)),
                importances=np.ones((15,)),
                fdr=0.8,
            )

    def test_selection_fdr_fdr_check(self, create_importances_pvalues):
        """Test selection fdr wrong"""
        importances, pvalues = create_importances_pvalues
        with pytest.raises(
            ValueError, match=r"'fdr' must be a float between 0 and 1, got -1."
        ):
            _selection_fdr(
                p_values=pvalues,
                importances=importances,
                fdr=-1,
            )
        with pytest.raises(
            ValueError,
            match=r"'fdr' must be a float between 0 and 1, got 1.1.",
        ):
            _selection_fdr(
                p_values=pvalues,
                importances=importances,
                fdr=1.1,
            )

    def test_selection_fdr_fdr_control_check(self, create_importances_pvalues):
        """Test selection fdr_control wrong"""
        importances, pvalues = create_importances_pvalues
        with pytest.raises(
            ValueError,
            match=r"'fdr_control' must be one of \('bhq', 'bhy'\), got wrong.",
        ):
            _selection_fdr(
                p_values=pvalues,
                importances=importances,
                fdr=0.8,
                fdr_control="wrong",
            )


class TestSelectionFWERParameterChecks:
    """Test class for 'selection_fwer' parameter checks"""

    def test_selection_fwer_pvalues_check(self, create_importances_pvalues):
        """Test selection pvalues wrong"""
        importances, _ = create_importances_pvalues
        with pytest.raises(
            ValueError, match=r"'p_values' must be an numpy ndarray"
        ):
            _selection_fwer(
                p_values=None,
                importances=importances,
                fwer=0.8,
            )
        with pytest.raises(
            ValueError, match=r"'p_values' must be an numpy ndarray"
        ):
            _selection_fwer(
                p_values=list(range(3)),
                importances=importances,
                fwer=0.8,
            )

    def test_selection_fwer_importances_check(
        self, create_importances_pvalues
    ):
        """Test selection importances wrong"""
        _, pvalues = create_importances_pvalues
        with pytest.raises(
            ValueError, match=r"'importances' must be an numpy ndarray"
        ):
            _selection_fwer(
                p_values=pvalues,
                importances=None,
                fwer=0.8,
            )
        with pytest.raises(
            ValueError, match=r"'importances' must be an numpy ndarray"
        ):
            _selection_fwer(
                p_values=pvalues,
                importances=list(range(3)),
                fwer=0.8,
            )

    def test_selection_fwer_pvalues_importances_shape_check(self):
        """Test selection pvalues and importances shape mismatch"""
        with pytest.raises(
            ValueError,
            match=r"Shape mismatch of 'p_values' and 'importances' on axis 0",
        ):
            _selection_fwer(
                p_values=np.ones((10,)),
                importances=np.ones((15,)),
                fwer=0.8,
            )

    def test_selection_fwer_fwer_check(self, create_importances_pvalues):
        """Test selection fwer wrong"""
        importances, pvalues = create_importances_pvalues
        with pytest.raises(
            ValueError,
            match=r"'fwer' must be a float between 0 and 1, got -1.",
        ):
            _selection_fwer(
                p_values=pvalues,
                importances=importances,
                fwer=-1,
            )
        with pytest.raises(
            ValueError,
            match=r"'fwer' must be a float between 0 and 1, got 1.1.",
        ):
            _selection_fwer(
                p_values=pvalues,
                importances=importances,
                fwer=1.1,
            )

    def test_selection_fwer_procedure_check(self, create_importances_pvalues):
        """Test selection procedure wrong"""
        importances, pvalues = create_importances_pvalues
        with pytest.raises(
            ValueError, match=r"Only 'bonferroni' procedure is supported."
        ):
            _selection_fwer(
                p_values=pvalues,
                importances=importances,
                fwer=0.8,
                procedure="wrong",
            )

    def test_selection_fwer_ntests_check(self, create_importances_pvalues):
        """Test selection n_tests wrong"""
        importances, pvalues = create_importances_pvalues
        with pytest.raises(
            ValueError, match=r"'n_tests' must be strictly positive."
        ):
            _selection_fwer(
                p_values=pvalues, importances=importances, fwer=0.8, n_tests=0
            )
