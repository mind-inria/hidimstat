import numpy as np
import pytest

from hidimstat.base_variable_importance import BaseVariableImportance


@pytest.fixture
def set_100_variable_sorted(seed):
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

    def test_selection_k_best(self, set_100_variable_sorted):
        "test selection of the k_best"
        vi = set_100_variable_sorted
        true_value = vi.importances_ >= 95
        selection = vi.importance_selection(k_best=5)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_k_best_none(self, set_100_variable_sorted):
        "test selection when there none"
        vi = set_100_variable_sorted
        true_value = np.ones_like(vi.importances_, dtype=bool)
        selection = vi.importance_selection(k_best=None)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_percentile(self, set_100_variable_sorted):
        "test selection bae on percentile"
        vi = set_100_variable_sorted
        true_value = vi.importances_ >= 50
        selection = vi.importance_selection(percentile=50)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_percentile_all(self, set_100_variable_sorted):
        "test selection when percentile is 100"
        vi = set_100_variable_sorted
        true_value = np.ones_like(vi.importances_, dtype=bool)
        true_value[np.argsort(vi.importances_)[0]] = False
        selection = vi.importance_selection(percentile=99.99)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_percentile_none(self, set_100_variable_sorted):
        "test selection when percentile is 0"
        vi = set_100_variable_sorted
        true_value = np.zeros_like(vi.importances_, dtype=bool)
        true_value[np.argsort(vi.importances_)[-1:]] = True
        selection = vi.importance_selection(percentile=0.1)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_percentile_threshols_value(self, set_100_variable_sorted):
        "test selection when percentile when the percentile equal on value"
        vi = set_100_variable_sorted
        mask = np.ones_like(vi.importances_, dtype=bool)
        mask[np.where(vi.importances_ == 99)] = False
        vi.importances_ = vi.importances_[mask]
        true_value = vi.importances_ >= 50
        selection = vi.importance_selection(percentile=50)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_threshold_min(self, set_100_variable_sorted):
        "test threshold minimal on importance"
        vi = set_100_variable_sorted
        true_value = vi.importances_ > 5
        selection = vi.importance_selection(threshold_min=5)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_threshold_max(self, set_100_variable_sorted):
        "test threshold maximal on importance"
        vi = set_100_variable_sorted
        true_value = vi.importances_ < 5
        selection = vi.importance_selection(threshold_max=5)
        np.testing.assert_array_equal(true_value, selection)


@pytest.mark.parametrize(
    "seed",
    [0],
    ids=["default_seed"],
)
class TestSelectionFDR:
    """Test selection based on fdr"""

    def test_selection_fdr_default(self, set_100_variable_sorted):
        "test selection of the default"
        vi = set_100_variable_sorted
        selection = vi.fdr_selection(0.2)
        assert np.all(
            [
                i >= (vi.importances_ - np.sum(selection))
                for i in vi.importances_[selection]
            ]
        )

    def test_selection_fdr_default_1(self, set_100_variable_sorted):
        "test selection of the default"
        vi = set_100_variable_sorted
        vi.pvalues_ = np.random.rand(vi.importances_.shape[0]) * 30
        true_value = np.zeros_like(vi.importances_, dtype=bool)  # selected any
        selection = vi.fdr_selection(0.2)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_fdr_bhy(self, set_100_variable_sorted):
        "test selection with bhy"
        vi = set_100_variable_sorted
        selection = vi.fdr_selection(0.2, fdr_control="bhy")
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
            vi.importance_selection()

    def test_selection_k_best(self, set_100_variable_sorted):
        "test selection k_best wrong"
        vi = set_100_variable_sorted
        with pytest.raises(AssertionError, match="k_best needs to be positive"):
            vi.importance_selection(k_best=-10)
        with pytest.warns(Warning, match="k=1000 is greater than n_features="):
            vi.importance_selection(k_best=1000)

    def test_selection_percentile(self, set_100_variable_sorted):
        "test selection percentile wrong"
        vi = set_100_variable_sorted
        with pytest.raises(
            AssertionError,
            match="percentile must be between 0 and 100 \(exclusive\). Got -1.",
        ):
            vi.importance_selection(percentile=-1)
        with pytest.raises(
            AssertionError,
            match="percentile must be between 0 and 100 \(exclusive\). Got 102.",
        ):
            vi.importance_selection(percentile=102)
        with pytest.raises(
            AssertionError,
            match="percentile must be between 0 and 100 \(exclusive\). Got 0.",
        ):
            vi.importance_selection(percentile=0)
        with pytest.raises(
            AssertionError,
            match="percentile must be between 0 and 100 \(exclusive\). Got 100",
        ):
            vi.importance_selection(percentile=100)

    def test_selection_pvalue_None(self, set_100_variable_sorted):
        "test selection on pvalue without it"
        vi = set_100_variable_sorted
        vi.pvalues_ = None
        with pytest.raises(
            AssertionError,
            match="The selection on p-value can't be done because the current method does not compute p-values.",
        ):
            vi.pvalue_selection(threshold_min=-1)

    def test_selection_threshold(self, set_100_variable_sorted):
        "test selection threshold wrong"
        vi = set_100_variable_sorted
        with pytest.raises(
            AssertionError, match="threshold_min needs to be between 0 and 1"
        ):
            vi.pvalue_selection(threshold_min=-1)
        with pytest.raises(
            AssertionError, match="threshold_min needs to be between 0 and 1"
        ):
            vi.pvalue_selection(threshold_min=1.1)
        with pytest.raises(
            AssertionError, match="threshold_max needs to be between 0 and 1"
        ):
            vi.pvalue_selection(threshold_max=-1)
        with pytest.raises(
            AssertionError, match="threshold_max needs to be between 0 and 1"
        ):
            vi.pvalue_selection(threshold_max=1.1)
        with pytest.raises(
            AssertionError,
            match="threshold_max needs to be higher than threshold_min",
        ):
            vi.pvalue_selection(threshold_max=0.5, threshold_min=0.9)


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
            vi.fdr_selection(0.1)

    def test_selection_fdr_pvalue_None(self, set_100_variable_sorted):
        "test selection fdr without pvalue"
        vi = set_100_variable_sorted
        vi.pvalues_ = None
        with pytest.raises(
            AssertionError,
            match="FDR-based selection requires p-values to be computed first. The current method does not support p-values.",
        ):
            vi.fdr_selection(fdr=0.1)

    def test_selection_fdr_fdr_control(self, set_100_variable_sorted):
        "test selection fdr_control wrong"
        vi = set_100_variable_sorted
        with pytest.raises(
            AssertionError,
            match="only 'bhq' and 'bhy' are supported",
        ):
            vi.fdr_selection(fdr=0.1, fdr_control="ehb")
