import pytest
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold

from hidimstat.base_variable_importance import BaseVariableImportance
from hidimstat.feature_selection.selectors import SelectorMixin


class TestSelectorMixinParameterCheck:
    """Class to test parameter checks of SelectorMixin"""

    def test_cv_parameter(self):
        selector = SelectorMixin(estimator=BaseVariableImportance())
        selector._check_fit_parameters()
        assert isinstance(selector.cv, StratifiedKFold)

        selector.cv = "wrong"
        with pytest.raises(
            TypeError, match=r"'cv' is not an sklearn-compatible"
        ):
            selector._check_fit_parameters()

    def test_estimator_parameter(self):
        selector = SelectorMixin(estimator=None)

        with pytest.raises(
            TypeError, match=r"'estimator' is not an instance of a hidimstat"
        ):
            selector._check_fit_parameters()

        selector.estimator = BaseEstimator()

        with pytest.raises(
            TypeError, match=r"'estimator' is not an instance of a hidimstat"
        ):
            selector._check_fit_parameters()
