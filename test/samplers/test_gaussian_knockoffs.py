import numpy as np
import pytest

from hidimstat._utils.scenario import multivariate_simulation
from hidimstat.samplers.gaussian_knockoffs import GaussianKnockoffs, _s_equi


def test_gaussian_equi():
    """test function of gaussian"""
    seed = 42
    n = 100
    p = 50
    rho = 0.5
    X, _, _, _ = multivariate_simulation(n, p, rho=rho, seed=seed)
    generator = GaussianKnockoffs()
    generator.fit(X=X)
    X_tilde = generator.sample(random_state=seed * 2)[0]
    assert X_tilde.shape == (n, p)
    assert np.linalg.norm(X - X_tilde) < np.sqrt(2) * np.linalg.norm(X)


def test_gaussian_error():
    """test function error"""
    seed = 42
    n = 100
    p = 50
    X, y, beta, noise = multivariate_simulation(n, p, seed=seed)
    generator = GaussianKnockoffs()
    with pytest.raises(
        ValueError, match="The GaussianGenerator requires to be fit before sampling"
    ):
        generator.sample(random_state=seed * 2)


def test_s_equi_not_definite_positive():
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
    gaussian_sampler = GaussianKnockoffs()
    gaussian_sampler.fit(X=X)
    X_tilde_1 = gaussian_sampler.sample(random_state=0)
    X_tilde_2 = gaussian_sampler.sample(random_state=0)
    assert np.array_equal(X_tilde_1, X_tilde_2)


def test_reproducibility_sample_repeat():
    """Test the repeatability of the samples"""
    X, _, _, _ = multivariate_simulation(100, 10, seed=0)
    gaussian_sampler = GaussianKnockoffs()
    gaussian_sampler.fit(X=X)
    X_tilde_1 = gaussian_sampler.sample(n_repeats=3, random_state=0)
    X_tilde_2 = gaussian_sampler.sample(n_repeats=3, random_state=0)
    assert np.array_equal(X_tilde_1, X_tilde_2)


def test_randomness_sample_no_seed():
    """Test the non repeatability of the samples when no seed"""
    X, _, _, _ = multivariate_simulation(100, 10, seed=0)
    gaussian_sampler = GaussianKnockoffs()
    gaussian_sampler.fit(X=X)
    X_tilde_1 = gaussian_sampler.sample(random_state=None)
    X_tilde_2 = gaussian_sampler.sample(random_state=None)
    assert not np.array_equal(X_tilde_1, X_tilde_2)


def test_randomness_sample_no_seed_repeat():
    """Test the non repeatability of the samples when no seed"""
    X, _, _, _ = multivariate_simulation(100, 10, seed=0)
    gaussian_sampler = GaussianKnockoffs()
    gaussian_sampler.fit(X=X)
    X_tilde_1 = gaussian_sampler.sample(n_repeats=3, random_state=None)
    X_tilde_2 = gaussian_sampler.sample(n_repeats=3, random_state=None)
    assert not np.array_equal(X_tilde_1, X_tilde_2)


def test_randomness_sample_rgn():
    """Test the non repeatability of the samples when the usage of random generator"""
    X, _, _, _ = multivariate_simulation(100, 10, seed=0)
    rng = np.random.default_rng(0)
    gaussian_sampler_rng = GaussianKnockoffs()
    gaussian_sampler_rng.fit(X=X)
    X_tilde_1 = gaussian_sampler_rng.sample(random_state=rng)
    X_tilde_2 = gaussian_sampler_rng.sample(random_state=rng)
    assert not np.array_equal(X_tilde_1, X_tilde_2)


def test_randomness_sample_rgn_repeat():
    """Test the non repeatability of the samples when the usage of random generator"""
    X, _, _, _ = multivariate_simulation(100, 10, seed=0)
    rng = np.random.default_rng(0)
    gaussian_sampler_rng = GaussianKnockoffs()
    gaussian_sampler_rng.fit(X=X)
    X_tilde_1 = gaussian_sampler_rng.sample(n_repeats=3, random_state=rng)
    X_tilde_2 = gaussian_sampler_rng.sample(n_repeats=3, random_state=rng)
    assert not np.array_equal(X_tilde_1, X_tilde_2)
