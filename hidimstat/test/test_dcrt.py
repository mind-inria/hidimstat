"""
Test the dcrt module
"""

from hidimstat.dcrt import dcrt_zero
from sklearn.datasets import make_regression


def test_dcrt_zero():
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.2)
    res = dcrt_zero(X, y, screening=False, verbose=True)
    assert len(res[0]) == 10
    assert len(res[1]) == 10
