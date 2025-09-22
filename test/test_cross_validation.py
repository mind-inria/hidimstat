import numpy as np
import pytest
from sklearn.model_selection import KFold

from hidimstat.base_variable_importance import BaseVariableImportance
from hidimstat.cross_validation import CrossValidation


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
def set_CrossValidationFDR(seed):
    nb_features = 100
    nb_samples = 200  # require to be higher that the factor + 10
    factor = 30
    rng = np.random.RandomState(seed)
    importance = np.arange(nb_features)
    rng.shuffle(importance)
    list_pvalues_generated = generate_list_pvalues_for_fdr(
        rng, importance, factor=factor
    )

    class MockVI(BaseVariableImportance):
        def fit(self, X, y):
            pass

        def importance(self, X, y):
            self.importances_ = [0]
            self.pvalues_ = [0]
            pass

    cv_mock_vi = CrossValidation(MockVI(), cv=KFold(factor + 9))
    cv_mock_vi.fit_importance(np.ones((nb_samples, nb_features)), np.ones(nb_samples))
    for index, vi in enumerate(cv_mock_vi.list_feature_importance_):
        vi.importances_ = importance
        vi.pvalues_ = importance  # list_pvalues_generated[index]
    return vi


@pytest.mark.parametrize(
    "seed",
    [0],
    ids=["default_seed"],
)
def test_CV(set_CrossValidationFDR):
    "test selection of the k_best"
    vi = set_CrossValidationFDR
    assert np.all(np.sort(vi.importances_) == np.arange(100))
    assert np.all(
        np.argsort(vi.pvalues_[np.argsort(vi.importances_)]) == np.arange(100)
    )
