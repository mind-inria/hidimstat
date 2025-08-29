import numpy as np
import pytest
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR

from hidimstat.statistical_tools.lasso_test import (
    preconfigure_LassoCV,
    lasso_statistic_with_sampling,
)


def test_preconfigure_LassoCV():
    """Test type errors"""
    with pytest.raises(
        TypeError, match="You should not use this function to configure the estimator"
    ):
        preconfigure_LassoCV(
            estimator=RidgeCV(),
            X=np.random.rand(10, 10),
            y=np.random.rand(10),
            X_tilde=np.random.rand(10, 10),
        )


def test_error_lasso_statistic_with_sampling_with_bad_config():
    """Test error lasso statistic"""
    with pytest.raises(
        TypeError, match="You should not use this function to configure the estimator"
    ):
        lasso_statistic_with_sampling(
            X=np.random.rand(10, 10),
            X_tilde=np.random.rand(10, 10),
            y=np.random.rand(10),
            lasso=SVR(),
        )


def test_error_lasso_statistic_with_sampling():
    """Test error lasso statistic"""
    with pytest.raises(TypeError, match="estimator should be linear"):
        lasso_statistic_with_sampling(
            X=np.random.rand(10, 10),
            X_tilde=np.random.rand(10, 10),
            y=np.random.rand(10),
            lasso=SVR(),
            preconfigure_lasso=None,
        )
