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
    X, _, _, _ = multivariate_simulation(n, p, seed=seed)
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
        ValueError, match="The GaussianGenerator requires to be fit before sampling"
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
    res = _s_equi(sigma)
    np.testing.assert_equal(res / np.diag(sigma), np.ones_like(res))


def test_reproducibility_sample():
    """Test the repeatability of the samples"""
    X, _, _, _ = multivariate_simulation(100, 10, seed=0)
    gaussian_sampler = GaussianKnockoffs(cov_estimator=LedoitWolf(), random_state=0)
    gaussian_sampler.fit(X=X)
    X_tilde_1 = gaussian_sampler.sample()
    X_tilde_2 = gaussian_sampler.sample()
    assert np.array_equal(X_tilde_1, X_tilde_2)


def test_randomness_sample_no_seed():
    """Test the non repeatability of the samples when no seed"""
    X, _, _, _ = multivariate_simulation(100, 10, seed=0)
    gaussian_sampler = GaussianKnockoffs(cov_estimator=LedoitWolf(), random_state=None)
    gaussian_sampler.fit(X=X)
    X_tilde_1 = gaussian_sampler.sample()
    X_tilde_2 = gaussian_sampler.sample()
    assert not np.array_equal(X_tilde_1, X_tilde_2)


def test_randomness_sample_rgn():
    """Test the non repeatability of the samples when the usage of random generator"""
    X, _, _, _ = multivariate_simulation(100, 10, seed=0)
    gaussian_sampler_rng = GaussianKnockoffs(
        cov_estimator=LedoitWolf(), random_state=np.random.RandomState(0)
    )
    gaussian_sampler_rng.fit(X=X)
    X_tilde_1 = gaussian_sampler_rng.sample()
    X_tilde_2 = gaussian_sampler_rng.sample()
    assert not np.array_equal(X_tilde_1, X_tilde_2)
