import numpy as np
import pytest
from hidimstat.statistical_tools.gaussian_knockoffs import (
    _s_equi,
    GaussianKnockoffs,
)
from hidimstat._utils.scenario import multivariate_simulation
from sklearn.covariance import LedoitWolf


def test_gaussian_equi():
    """test function of gaussian"""
    seed = 42
    n = 100
    p = 50
    X, y, beta, noise = multivariate_simulation(n, p, seed=seed)
    generator = GaussianKnockoffs(
        cov_estimator=LedoitWolf(),
        random_state=seed * 2,
    )
    generator.fit(X=X)
    X_tilde = generator.sample()
    assert X_tilde.shape == (n, p)


def test_gaussian_error():
    """test function error"""
    seed = 42
    n = 100
    p = 50
    X, y, beta, noise = multivariate_simulation(n, p, seed=seed)
    generator = GaussianKnockoffs(
        cov_estimator=LedoitWolf(),
        random_state=seed * 2,
    )
    with pytest.raises(
        ValueError, match="The GaussianGenerator requires to be fit before simulate"
    ):
        generator.sample()


def test_s_equi_not_define_positive():
    """test the warning and error of s_equi function"""
    n = 10
    tol = 1e-7
    seed = 42

    # random positive matrix
    rgn = np.random.RandomState(seed)
    a = rgn.randn(n, n)
    a -= np.min(a)
    with pytest.raises(
        Exception, match="The covariance matrix is not positive-definite."
    ):
        _s_equi(a)

    # matrix with positive eigenvalues, positive diagonal
    while not np.all(np.linalg.eigvalsh(a) > tol):
        a += 0.1 * np.eye(n)
    with pytest.warns(UserWarning, match="The equi-correlated matrix"):
        _s_equi(a)

    # positive definite matrix
    u, s, vh = np.linalg.svd(a)
    d = np.eye(n)
    sigma = u * d * u.T
    _s_equi(sigma)
