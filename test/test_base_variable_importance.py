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
        selection = vi.selection(k_best=5)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_k_best_none(self, set_100_variable_sorted):
        "test selection when there none"
        vi = set_100_variable_sorted
        true_value = np.ones_like(vi.importances_, dtype=bool)
        selection = vi.selection(k_best=None)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_percentile(self, set_100_variable_sorted):
        "test selection bae on percentile"
        vi = set_100_variable_sorted
        true_value = vi.importances_ >= 50
        selection = vi.selection(percentile=50)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_percentile_all(self, set_100_variable_sorted):
        "test selection when percentile is 100"
        vi = set_100_variable_sorted
        true_value = np.ones_like(vi.importances_, dtype=bool)
        true_value[np.argsort(vi.importances_)[0]] = False
        selection = vi.selection(percentile=99.99)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_percentile_none(self, set_100_variable_sorted):
        "test selection when percentile is 0"
        vi = set_100_variable_sorted
        true_value = np.zeros_like(vi.importances_, dtype=bool)
        true_value[np.argsort(vi.importances_)[-1:]] = True
        selection = vi.selection(percentile=0.1)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_percentile_threshols_value(self, set_100_variable_sorted):
        "test selection when percentile when the percentile equal on value"
        vi = set_100_variable_sorted
        mask = np.ones_like(vi.importances_, dtype=bool)
        mask[np.where(vi.importances_ == 99)] = False
        vi.importances_ = vi.importances_[mask]
        true_value = vi.importances_ >= 50
        selection = vi.selection(percentile=50)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_threshold(self, set_100_variable_sorted):
        "test threshold on importance"
        vi = set_100_variable_sorted
        true_value = vi.importances_ < 5
        selection = vi.selection(threshold=5)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_threshold_pvalue(self, set_100_variable_sorted):
        "test threshold vbse on pvalues"
        vi = set_100_variable_sorted
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

    def test_selection_fdr_default(self, set_100_variable_sorted):
        "test selection of the default"
        vi = set_100_variable_sorted
        selection = vi.selection_fdr(0.2)
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
        if hasattr(vi, "list_pvalues_"):
            vi.list_pvalues_ = [
                np.random.rand(vi.importances_.shape[0]) * 30 for i in range(10)
            ]
        true_value = np.zeros_like(vi.importances_, dtype=bool)  # selected any
        selection = vi.selection_fdr(0.2)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_fdr_bhy(self, set_100_variable_sorted):
        "test selection with bhy"
        vi = set_100_variable_sorted
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

    def test_selection_k_best(self, set_100_variable_sorted):
        "test selection k_best wrong"
        vi = set_100_variable_sorted
        with pytest.raises(AssertionError, match="k_best needs to be positive"):
            vi.selection(k_best=-10)
        with pytest.warns(Warning, match="k=1000 is greater than n_features="):
            vi.selection(k_best=1000)

    def test_selection_percentile(self, set_100_variable_sorted):
        "test selection percentile wrong"
        vi = set_100_variable_sorted
        with pytest.raises(
            AssertionError,
            match="percentile must be between 0 and 100 \(exclusive\). Got -1.",
        ):
            vi.selection(percentile=-1)
        with pytest.raises(
            AssertionError,
            match="percentile must be between 0 and 100 \(exclusive\). Got 102.",
        ):
            vi.selection(percentile=102)
        with pytest.raises(
            AssertionError,
            match="percentile must be between 0 and 100 \(exclusive\). Got 0.",
        ):
            vi.selection(percentile=0)
        with pytest.raises(
            AssertionError,
            match="percentile must be between 0 and 100 \(exclusive\). Got 100",
        ):
            vi.selection(percentile=100)

    def test_selection_threshold(self, set_100_variable_sorted):
        "test selection threshold wrong"
        vi = set_100_variable_sorted
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

    def test_selection_fdr_fdr_control(self, set_100_variable_sorted):
        "test selection fdr_control wrong"
        vi = set_100_variable_sorted
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
