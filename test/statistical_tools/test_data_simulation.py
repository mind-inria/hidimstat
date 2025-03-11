import numpy as np
from hidimstat._utils.scenario import multivariate_1D_simulation_AR


def test_simu_data():
    X, y, _, _ = multivariate_1D_simulation_AR(n_samples=100, n_features=200, seed=42)

    assert X.shape == (100, 200)
    assert y.size == 100


def test_non_zero_index():
    # Test to make sure non-null variable indices are sampled without
    # replacement
    X, y, _, non_zero = multivariate_1D_simulation_AR(10, 10, sparsity=1.0, seed=0)
    assert non_zero.size == np.unique(non_zero).size
