import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from hidimstat.base_perturbation import BasePerturbation


def test_no_implemented_methods():
    """test that the methods are not implemented in the base class"""
    X = np.random.randint(0, 2, size=(100, 2, 1))
    estimator = LinearRegression()
    estimator.fit(X[:, 0], X[:, 1])
    basic_class = BasePerturbation(estimator=estimator)
    with pytest.raises(NotImplementedError):
        basic_class._permutation(X, features_group_id=None)
