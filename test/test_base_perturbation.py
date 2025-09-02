from hidimstat.base_perturbation import BasePerturbation
from sklearn.linear_model import LinearRegression
import pytest
import numpy as np


def test_no_implemented_methods():
    """test that the methods are not implemented in the base class"""
    X = np.random.randint(0, 2, size=(100, 2, 1))
    estimator = LinearRegression()
    estimator.fit(X[:, 0], X[:, 1])
    basic_class = BasePerturbation(estimator=estimator)
    with pytest.raises(NotImplementedError):
        basic_class._permutation(X, group_id=None)


def test_chek_importance():
    """test that the methods are not implemented in the base class"""
    X = np.random.randint(0, 2, size=(100, 2, 1))
    estimator = LinearRegression()
    estimator.fit(X[:, 0], X[:, 1])
    basic_class = BasePerturbation(estimator=estimator)
    basic_class.importances_ = []
    with pytest.raises(
        ValueError, match="The importances need to be called before calling this method"
    ):
        basic_class.selection()
