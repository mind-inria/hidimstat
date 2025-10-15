from functools import partial

import numpy as np
import pytest
from scipy.stats import ttest_1samp, wilcoxon

from hidimstat._utils.utils import (
    check_random_state,
    check_statistical_test,
    get_fitted_attributes,
)


def test_generated_attributes():
    """Test function for getting generated attribute"""

    class MyClass:
        def __init__(self):
            self.attr1 = 1
            self.attr2_ = 2
            self._attr3 = 3
            self.attr4__ = 4
            self.attr5_ = 5

    attributes = get_fitted_attributes(MyClass())
    assert attributes == ["attr2_", "attr5_"]


def test_none():
    "test random state is None"
    random_state = None
    rng = check_random_state(random_state)
    assert isinstance(rng, np.random.Generator)


def test_integer():
    "test random state is integer"
    random_state = 10
    rng = check_random_state(random_state)
    assert isinstance(rng, np.random.Generator)


def test_rng():
    "test random state is rng"
    rng = np.random.default_rng(0)
    assert isinstance(rng, np.random.Generator)


def test_random_state():
    "test random state is RandomState"
    random_state = np.random.RandomState(0)
    with pytest.raises(
        ValueError,
        match="numpy.random.RandomState is deprecated. Please use numpy.random.Generator",
    ):
        _ = check_random_state(random_state)


def test_error():
    "test random state is rng"
    random_state = [1, 2, 3]
    with pytest.raises(
        ValueError, match="cannot be used to seed a numpy.random.Generator instance"
    ):
        check_random_state(random_state)


def test_check_test_statistic():
    "test the function of check"
    test_func = check_statistical_test("wilcoxon")
    assert test_func.func == wilcoxon
    test_func = check_statistical_test("ttest")
    assert test_func.func == ttest_1samp
    test_func = check_statistical_test(print)
    assert test_func == print


def test_check_test_statistic_warning():
    "test the exception"
    with pytest.raises(ValueError, match="the test 'test' is not supported"):
        check_statistical_test("test")
    with pytest.raises(ValueError, match="is not a valid test"):
        check_statistical_test([])
