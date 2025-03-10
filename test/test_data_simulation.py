import numpy as np
from hidimstat._utils.data_simulation import simu_data


def test_simu_data():
    X, y, _, _ = simu_data(n=100, p=200, seed=42)

    assert X.shape == (100, 200)
    assert y.size == 100


def test_non_zero_index():
    # Test to make sure non-null variable indices are sampled without
    # replacement
    X, y, _, non_zero = simu_data(10, 10, sparsity=1.0, seed=0)
    assert non_zero.size == np.unique(non_zero).size
