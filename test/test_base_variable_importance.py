import numpy as np
import pytest

from hidimstat.base_variable_importance import BaseVariableImportance


def generate_list_pvalues_for_fdr(rng, importances, factor=30):
    """Generate values for applying FDR.
    TODO: Improve data generation.
    """
    nb_features = importances.shape[0]
    result_list = []
    for i in range(10):
        score = rng.rand(nb_features) * factor
        result_list.append(score)
    for i in range(1, factor):
        score = rng.rand(nb_features) + 1
        score[-i:] = np.arange(factor - i, factor) * 2
        score[:i] = -np.arange(factor - i, factor)
        result_list.append(np.flip(score)[importances])
    return np.array(result_list) / np.max(result_list)


@pytest.fixture
def set_BaseVariableImportance(pvalues, list_pvalues, seed):
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
    nb_features = 100
    rng = np.random.RandomState(seed)
    vi = BaseVariableImportance()
    vi.importances_ = np.arange(nb_features)
    rng.shuffle(vi.importances_)
    list_pvalues_generated = generate_list_pvalues_for_fdr(rng, vi.importances_)
    if pvalues or list_pvalues:
        vi.pvalues_ = np.mean(list_pvalues_generated, axis=0)
    if list_pvalues:
        vi.list_pvalues_ = list_pvalues_generated
    return vi


@pytest.mark.parametrize(
    "pvalues, list_pvalues, seed",
    [(False, False, 0), (True, False, 1), (True, True, 2)],
    ids=["only importance", "p-value", "list_pvalues"],
)
class TestSelection:
    """Test selection based on importance"""

    def test_selection_k_best(self, set_BaseVariableImportance):
        "test selection of the k_best"
        vi = set_BaseVariableImportance
        true_value = vi.importances_ >= 95
        selection = vi.selection(k_best=5)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_k_best_all(self, set_BaseVariableImportance):
        "test selection to all base on string"
        vi = set_BaseVariableImportance
        true_value = np.ones_like(vi.importances_, dtype=bool)
        selection = vi.selection(k_best="all")
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_k_best_none(self, set_BaseVariableImportance):
        "test selection when there none"
        vi = set_BaseVariableImportance
        true_value = np.zeros_like(vi.importances_, dtype=bool)
        selection = vi.selection(k_best=0)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_percentile(self, set_BaseVariableImportance):
        "test selection bae on percentile"
        vi = set_BaseVariableImportance
        true_value = vi.importances_ >= 50
        selection = vi.selection(percentile=50)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_percentile_all(self, set_BaseVariableImportance):
        "test selection when percentile is 100"
        vi = set_BaseVariableImportance
        true_value = np.ones_like(vi.importances_, dtype=bool)
        selection = vi.selection(percentile=100)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_percentile_none(self, set_BaseVariableImportance):
        "test selection when percentile is 0"
        vi = set_BaseVariableImportance
        true_value = np.zeros_like(vi.importances_, dtype=bool)
        selection = vi.selection(percentile=0)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_percentile_threshols_value(self, set_BaseVariableImportance):
        "test selection when percentile when the percentile equal on value"
        vi = set_BaseVariableImportance
        mask = np.ones_like(vi.importances_, dtype=bool)
        mask[np.where(vi.importances_ == 99)] = False
        vi.importances_ = vi.importances_[mask]
        true_value = vi.importances_ >= 50
        selection = vi.selection(percentile=50)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_threshold(self, set_BaseVariableImportance):
        "test threshold on importance"
        vi = set_BaseVariableImportance
        true_value = vi.importances_ < 5
        selection = vi.selection(threshold=5)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_threshold_pvalue(self, set_BaseVariableImportance):
        "test threshold vbse on pvalues"
        vi = set_BaseVariableImportance
        if vi.pvalues_ is not None:
            true_value = vi.importances_ > 5
            selection = vi.selection(
                threshold_pvalue=vi.pvalues_[np.argsort(vi.importances_)[5]]
            )
            np.testing.assert_array_equal(true_value, selection)


@pytest.mark.parametrize(
    "pvalues, list_pvalues, seed",
    [(True, False, 10), (True, True, 10)],
    ids=["pvalue_only", "list_pvalue"],
)
class TestSelectionFDR:
    """Test selection based on fdr"""

    def test_selection_fdr_default(self, set_BaseVariableImportance):
        "test selection of the default"
        vi = set_BaseVariableImportance
        selection = vi.selection_fdr(0.2)
        assert np.all(
            [
                i >= (vi.importances_ - np.sum(selection))
                for i in vi.importances_[selection]
            ]
        )

    def test_selection_fdr_default_1(self, set_BaseVariableImportance):
        "test selection of the default"
        vi = set_BaseVariableImportance
        vi.pvalues_ = np.random.rand(vi.importances_.shape[0]) * 30
        if hasattr(vi, "list_pvalues_"):
            vi.list_pvalues_ = [
                np.random.rand(vi.importances_.shape[0]) * 30 for i in range(10)
            ]
        true_value = np.zeros_like(vi.importances_, dtype=bool)  # selected any
        selection = vi.selection_fdr(0.2)
        np.testing.assert_array_equal(true_value, selection)

    def test_selection_fdr_adaptation(self, set_BaseVariableImportance):
        "test selection of the adaptation"
        vi = set_BaseVariableImportance
        selection = vi.selection_fdr(0.2, adaptive_aggregation=True)
        assert np.all(
            [
                i >= (vi.importances_ - np.sum(selection))
                for i in vi.importances_[selection]
            ]
        )

    def test_selection_fdr_bhy(self, set_BaseVariableImportance):
        "test selection with bhy"
        vi = set_BaseVariableImportance
        selection = vi.selection_fdr(0.2, fdr_control="bhy")
        assert np.all(
            [
                i >= (vi.importances_ - np.sum(selection))
                for i in vi.importances_[selection]
            ]
        )


@pytest.mark.parametrize(
    "pvalues, list_pvalues, seed",
    [(False, False, 0), (True, False, 0), (True, True, 0)],
    ids=["only importance", "p-value", "list_pvalues"],
)
class TestBVIExceptions:
    """Test class for BVI Exception"""

    def test_not_fit(self, pvalues, list_pvalues, seed):
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
        with pytest.raises(
            ValueError,
            match="The importances need to be called before calling this method",
        ):
            vi.selection_fdr(0.1)

    def test_selection_k_best(self, set_BaseVariableImportance):
        "test selection k_best wrong"
        vi = set_BaseVariableImportance
        with pytest.raises(AssertionError, match="k_best needs to be positive or null"):
            vi.selection(k_best=-10)
        with pytest.warns(Warning, match="k=1000 is greater than n_features="):
            vi.selection(k_best=1000)

    def test_selection_percentile(self, set_BaseVariableImportance):
        "test selection percentile wrong"
        vi = set_BaseVariableImportance
        with pytest.raises(
            AssertionError, match="percentile needs to be between 0 and 100"
        ):
            vi.selection(percentile=-1)
        with pytest.raises(
            AssertionError, match="percentile needs to be between 0 and 100"
        ):
            vi.selection(percentile=102)

    def test_selection_threshold(self, set_BaseVariableImportance):
        "test selection threshold wrong"
        vi = set_BaseVariableImportance
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

    def test_selection_fdr_fdr_control(self, set_BaseVariableImportance):
        "test selection fdr_control wrong"
        vi = set_BaseVariableImportance
        if vi.pvalues_ is None:
            with pytest.raises(
                AssertionError,
                match="this method doesn't support selection base on FDR",
            ):
                vi.selection_fdr(fdr=0.1)
        else:
            with pytest.raises(
                AssertionError,
                match="only 'bhq' and 'bhy' are supported",
            ):
                vi.selection_fdr(fdr=0.1, fdr_control="ehb")
