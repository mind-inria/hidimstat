import pytest
from sklearn.base import BaseEstimator

from hidimstat.base_perturbation import BasePerturbation
from hidimstat.selector import SelectorMixin


class TestSelectorMixinParameterCheck:
    """Class to test parameter checks of SelectorMixin"""

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

        selector.estimator = BasePerturbation()

        with pytest.raises(
            TypeError, match=r"Instances of BasePerturbation are not supported"
        ):
            selector._check_fit_parameters()
