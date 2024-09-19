import numpy as np
import pytest


@pytest.fixture
def linear_scenario():
    X = np.random.randn(100, 10)
    beta = np.zeros(10)
    important_features = np.random.choice(10, 2, replace=False)
    beta[important_features] = 5
    y = X @ beta
    return X, y, beta
