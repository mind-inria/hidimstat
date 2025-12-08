import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from hidimstat.base_perturbation import BasePerturbation, BasePerturbationCV


def test_no_implemented_methods():
    """test that the methods are not implemented in the base class"""
    X = np.random.randint(0, 2, size=(100, 2, 1))
    estimator = LinearRegression()
    estimator.fit(X[:, 0], X[:, 1])
    basic_class = BasePerturbation(estimator=estimator)
    with pytest.raises(NotImplementedError):
        basic_class._permutation(X, features_group_id=None)


def test_check_importance():
    """test that the methods are not implemented in the base class"""
    X = np.random.randint(0, 2, size=(100, 2, 1))
    estimator = LinearRegression()
    estimator.fit(X[:, 0], X[:, 1])
    basic_class = BasePerturbation(estimator=estimator)
    basic_class.importances_ = []
    with pytest.raises(
        ValueError, match="The importance method has not yet been called."
    ):
        basic_class.importance_selection()


def test_base_cv_errors():
    """Test the errors of the BasePerturbationCV class"""
    with pytest.raises(
        ValueError,
        match="If estimators is a list, its length must be equal to the number of folds",
    ):
        BasePerturbationCV(
            estimators=[LinearRegression(), LinearRegression()], cv=KFold(n_splits=4)
        )

    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    with pytest.raises(NotImplementedError):
        vim = BasePerturbationCV(estimators=LinearRegression(), cv=KFold(n_splits=2))
        vim.fit(X, y)
