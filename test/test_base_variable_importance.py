import numpy as np
import pytest

from hidimstat.base_variable_importance import BaseVariableImportance


@pytest.fixture
def _set_variable_importance(seed):
    """Create a BaseVariableImportance instance with test data for testing purposes.

    Parameters
    ----------
    pvalues : bool
        If True, generate random p-values for testing.
    test_score : bool
        If True, generate random test scores for testing.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    BaseVariableImportance
        A BaseVariableImportance instance with test data.
    """
    n_features = 100
    rng = np.random.RandomState(seed)
    vi = BaseVariableImportance()
    vi.importances_ = np.arange(n_features)
    rng.shuffle(vi.importances_)
    vi.pvalues_ = np.flip(np.sort(rng.random(n_features)))[vi.importances_]
    return vi


@pytest.mark.parametrize(
    "seed",
    [0, 2],
    ids=["default_seed", "another seed"],
)
class TestSelection:
    """Test selection based on importance"""

    def test_selection_k_best(self, _set_variable_importance):
        "test selection of the k_best"
        vi = _set_variable_importance
        true_value = vi.importances_ >= 95
        selection = vi.selection(k_best=5)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_k_best_all(self, _set_variable_importance):
        "test selection to all base on string"
        vi = _set_variable_importance
        true_value = np.ones_like(vi.importances_, dtype=bool)
        selection = vi.selection(k_best="all")
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_k_best_none(self, _set_variable_importance):
        "test selection when there none"
        vi = _set_variable_importance
        true_value = np.zeros_like(vi.importances_, dtype=bool)
        selection = vi.selection(k_best=0)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_percentile(self, _set_variable_importance):
        "test selection bae on percentile"
        vi = _set_variable_importance
        true_value = vi.importances_ >= 50
        selection = vi.selection(percentile=50)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_percentile_all(self, _set_variable_importance):
        "test selection when percentile is 100"
        vi = _set_variable_importance
        true_value = np.ones_like(vi.importances_, dtype=bool)
        selection = vi.selection(percentile=100)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_percentile_none(self, _set_variable_importance):
        "test selection when percentile is 0"
        vi = _set_variable_importance
        true_value = np.zeros_like(vi.importances_, dtype=bool)
        selection = vi.selection(percentile=0)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_percentile_threshols_value(self, _set_variable_importance):
        "test selection when percentile when the percentile equal on value"
        vi = _set_variable_importance
        mask = np.ones_like(vi.importances_, dtype=bool)
        mask[np.where(vi.importances_ == 99)] = False
        vi.importances_ = vi.importances_[mask]
        true_value = vi.importances_ >= 50
        selection = vi.selection(percentile=50)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_threshold(self, _set_variable_importance):
        "test threshold on importance"
        vi = _set_variable_importance
        true_value = vi.importances_ < 5
        selection = vi.selection(threshold=5)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_threshold_pvalue(self, _set_variable_importance):
        "test threshold vbse on pvalues"
        vi = _set_variable_importance
        true_value = vi.importances_ > 5
        selection = vi.selection(
            threshold_pvalue=vi.pvalues_[np.argsort(vi.importances_)[5]]
        )
        np.testing.assert_array_equal(true_value, selection)


@pytest.mark.parametrize(
    "seed",
    [0],
    ids=["default_seed"],
)
class TestSelectionFDR:
    """Test selection based on fdr"""

    def test_selection_fdr_default(self, _set_variable_importance):
        "test selection of the default"
        vi = _set_variable_importance
        selection = vi.selection_fdr(0.2)
        assert np.all(
            [
                i >= (vi.importances_ - np.sum(selection))
                for i in vi.importances_[selection]
            ]
        )

    def test_selection_fdr_default_1(self, _set_variable_importance):
        "test selection of the default"
        vi = _set_variable_importance
        vi.pvalues_ = np.random.rand(vi.importances_.shape[0]) * 30
        if hasattr(vi, "list_pvalues_"):
            vi.list_pvalues_ = [
                np.random.rand(vi.importances_.shape[0]) * 30 for i in range(10)
            ]
        true_value = np.zeros_like(vi.importances_, dtype=bool)  # selected any
        selection = vi.selection_fdr(0.2)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_fdr_bhy(self, _set_variable_importance):
        "test selection with bhy"
        vi = _set_variable_importance
        selection = vi.selection_fdr(0.2, fdr_control="bhy")
        assert np.all(
            [
                i >= (vi.importances_ - np.sum(selection))
                for i in vi.importances_[selection]
            ]
        )


@pytest.mark.parametrize(
    "seed",
    [0],
    ids=["default_seed"],
)
class TestBVIExceptions:
    """Test class for BVI Exception"""

    def test_not_fit(self, seed):
        "test detection unfit"
        vi = BaseVariableImportance()
        with pytest.raises(
            ValueError,
            match="The importances need to be called before calling this method",
        ):
            vi._check_importance()
        with pytest.raises(
            ValueError,
            match="The importances need to be called before calling this method",
        ):
            vi.selection()

    def test_selection_k_best(self, _set_variable_importance):
        "test selection k_best wrong"
        vi = _set_variable_importance
        with pytest.raises(AssertionError, match="k_best needs to be positive or null"):
            vi.selection(k_best=-10)
        with pytest.warns(Warning, match="k=1000 is greater than n_features="):
            vi.selection(k_best=1000)

    def test_selection_percentile(self, _set_variable_importance):
        "test selection percentile wrong"
        vi = _set_variable_importance
        with pytest.raises(
            AssertionError, match="percentile needs to be between 0 and 100"
        ):
            vi.selection(percentile=-1)
        with pytest.raises(
            AssertionError, match="percentile needs to be between 0 and 100"
        ):
            vi.selection(percentile=102)

    def test_selection_threshold(self, _set_variable_importance):
        "test selection threshold wrong"
        vi = _set_variable_importance
        if vi.pvalues_ is None:
            with pytest.raises(
                AssertionError,
                match="This method doesn't support a threshold on p-values",
            ):
                vi.selection(threshold_pvalue=-1)
        else:
            with pytest.raises(
                AssertionError, match="threshold_pvalue needs to be between 0 and 1"
            ):
                vi.selection(threshold_pvalue=-1)
            with pytest.raises(
                AssertionError, match="threshold_pvalue needs to be between 0 and 1"
            ):
                vi.selection(threshold_pvalue=1.1)


@pytest.mark.parametrize(
    "seed",
    [0],
    ids=["default_seed"],
)
class TestSelectionFDRExceptions:
    def test_not_fit(self, seed):
        "test detection unfit"

        vi = BaseVariableImportance()
        with pytest.raises(
            ValueError,
            match="The importances need to be called before calling this method",
        ):
            vi.selection_fdr(0.1)

    def test_selection_fdr_fdr_control(self, _set_variable_importance):
        "test selection fdr_control wrong"
        vi = _set_variable_importance
        if vi.pvalues_ is None:
            with pytest.raises(
                TypeError,
                match="object of type 'NoneType' has no len()",
            ):
                vi.selection_fdr(fdr=0.1)
        else:
            with pytest.raises(
                AssertionError,
                match="only 'bhq' and 'bhy' are supported",
            ):
                vi.selection_fdr(fdr=0.1, fdr_control="ehb")
