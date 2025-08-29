import numpy as np
import pytest
from sklearn.svm import SVR

from hidimstat.statistical_tools.lasso_test import (
    lasso_statistic,
)


def test_error_lasso_statistic():
    """Test error lasso statistic"""
    with pytest.raises(TypeError, match="estimator should be linear"):
        lasso_statistic(X=np.random.rand(10, 10), y=np.random.rand(10), lasso=SVR())
