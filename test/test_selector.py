import inspect

import pytest
from sklearn.base import BaseEstimator

from hidimstat import PFI, PFICV
from hidimstat.selector import SelectorMixin


class TestSelectorMixinParameterCheck:
    """Class to test parameter checks of SelectorMixin"""

    def test_estimator_parameter(self):
        """Test estimator parameter checks"""
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

        pfi = PFI()
        selector.estimator = pfi

        with pytest.warns(
            UserWarning, match=r"An instance of BasePerturbation was given"
        ):
            selector._check_fit_parameters()
            assert isinstance(selector.estimator, PFICV)

            # Now we check that the fallback from PFI to PFICV worked
            # Get all init parameters of PFI
            signature = inspect.signature(pfi.__class__).parameters
            pfi_init_params = {}
            for name in signature:
                pfi_init_params[name] = getattr(pfi, name)

            # Get all init parameters of PFICV
            signature = inspect.signature(
                selector.estimator.__class__
            ).parameters
            pficv_init_params = {}
            for name in signature:
                pficv_init_params[name] = getattr(selector.estimator, name)

            # Make sure that all parameters from PFI and PFICV are equal
            assert (
                pfi_init_params["estimator"] == pficv_init_params["estimators"]
            )
            del pfi_init_params["estimator"]
            del pficv_init_params["estimators"]

            for name in pfi_init_params:
                assert pfi_init_params[name] == pficv_init_params[name]
