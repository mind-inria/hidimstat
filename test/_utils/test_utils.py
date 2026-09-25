import numpy as np
import pytest
from scipy.stats import ttest_1samp, wilcoxon
from sklearn.linear_model import (
    LassoCV,
    LogisticRegressionCV,
    Ridge,
    RidgeClassifier,
)
from sklearn.metrics import (
    get_scorer,
    log_loss,
    make_scorer,
    mean_squared_error,
)

from hidimstat._utils.utils import (
    SKLEARN_LT_1_6,
    _make_sklearn_estimator,
    check_random_state,
    check_scoring,
    check_statistical_test,
    find_stack_level,
    get_fitted_attributes,
    one_level_deeper,
)
from hidimstat.statistical_tools import nadeau_bengio_ttest


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
    """Test random state is None"""
    random_state = None
    rng = check_random_state(random_state)
    assert isinstance(rng, np.random.Generator)


def test_integer():
    """Test random state is integer"""
    random_state = 10
    rng = check_random_state(random_state)
    assert isinstance(rng, np.random.Generator)


def test_rng():
    """Test random state is rng"""
    rng = np.random.default_rng(0)
    assert isinstance(rng, np.random.Generator)


def test_random_state():
    """Test random state is RandomState"""
    random_state = np.random.RandomState(0)
    with pytest.raises(
        ValueError,
        match=r"numpy\.random\.RandomState is deprecated\. Please use numpy\.random.Generator",
    ):
        _ = check_random_state(random_state)


def test_error():
    """Test random state is rng"""
    random_state = [1, 2, 3]
    with pytest.raises(
        ValueError,
        match=r"cannot be used to seed a numpy\.random\.Generator instance",
    ):
        check_random_state(random_state)


def test_check_test_statistic():
    """Test the function of check"""
    test_func = check_statistical_test("wilcoxon")
    assert test_func.func == wilcoxon
    test_func = check_statistical_test("ttest")
    assert test_func.func == ttest_1samp
    test_func = check_statistical_test("nb-ttest")
    assert test_func.func == nadeau_bengio_ttest
    test_func = check_statistical_test(print)
    assert test_func == print
    test_func = check_statistical_test(lambda x: x)
    assert test_func.__class__.__name__ == "function"


def test_check_test_statistic_warning():
    """Test the exception"""
    with pytest.raises(ValueError, match="the test 'test' is not supported"):
        check_statistical_test("test")
    with pytest.raises(
        ValueError, match="Unsupported value for 'statistical_test'"
    ):
        check_statistical_test([])


def test__make_sklearn_estimator(monkeypatch):

    monkeypatch.setattr("hidimstat._utils.utils.SKLEARN_LT_1_9", True)

    for penalty, expected in zip(["l1", "l2"], [(1,), (0,)], strict=False):
        est = _make_sklearn_estimator(
            LogisticRegressionCV,
            penalty=penalty,
        )

        assert est.l1_ratios == expected

    monkeypatch.setattr("hidimstat._utils.utils.SKLEARN_LT_1_9", False)

    for l1_ratio, expected in zip([(1,), (0,)], ["l1", "l2"], strict=False):
        est = _make_sklearn_estimator(
            LogisticRegressionCV,
            l1_ratios=l1_ratio,
        )

        assert est.penalty == expected

    target = 10
    if SKLEARN_LT_1_6:
        est = _make_sklearn_estimator(
            LassoCV,
            alphas=target,
        )
        assert est.n_alphas == target
    else:
        est = _make_sklearn_estimator(
            LassoCV,
            n_alphas=target,
        )
        assert est.alphas == target

    est = _make_sklearn_estimator(LassoCV, n_alphas=target)

    assert est.alphas == 10


def test_find_stack_level():
    """Test find_stack_level."""
    assert find_stack_level() == 1
    assert one_level_deeper() == 2


def test_check_scoring():
    """Test the in-house check_scoring function"""
    regressor_scorer = repr(get_scorer(make_scorer(mean_squared_error)))
    assert (
        repr(check_scoring(scoring="mean_squared_error")) == regressor_scorer
    )
    assert (
        repr(check_scoring(scoring=make_scorer(mean_squared_error)))
        == regressor_scorer
    )
    assert repr(check_scoring(estimator=Ridge())) == regressor_scorer

    classifier_scorer = repr(
        get_scorer(make_scorer(log_loss, response_method="predict_proba"))
    )
    assert repr(check_scoring(scoring="log_loss")) == classifier_scorer
    assert (
        repr(
            check_scoring(
                scoring=make_scorer(log_loss, response_method="predict_proba")
            )
        )
        == classifier_scorer
    )
    assert (
        repr(check_scoring(estimator=RidgeClassifier())) == classifier_scorer
    )
