# -*- coding: utf-8 -*-
# Authors: Binh Nguyen <tuan-binh.nguyen@inria.fr>

from hidimstat.data_simulation import simu_data
from hidimstat.gaussian_knockoff import (
    _estimate_distribution,
    gaussian_knockoff_generation,
)
import pytest



def test_estimate_distribution_wolf():
    SEED = 42
    n = 100
    p = 50
    X, y, _, non_zero = simu_data(n, p, seed=SEED)
    mu, Sigma = _estimate_distribution(X, cov_estimator="ledoit_wolf")

    assert mu.size == p
    assert Sigma.shape == (p, p)


def test_estimate_distribution_lasso():
    SEED = 42
    n = 100
    p = 50
    X, y, _, non_zero = simu_data(n, p, seed=SEED)
    mu, Sigma = _estimate_distribution(X, shrink=True, cov_estimator="graph_lasso")

    assert mu.size == p
    assert Sigma.shape == (p, p)

def test_gaussian_knockoff_estimate_exception():
    SEED = 42
    n = 100
    p = 50
    X, y, _, non_zero = simu_data(n, p, seed=SEED)
    with pytest.raises(ValueError, match="test is not a valid covariance estimated method"):
        _estimate_distribution(X, shrink=True, cov_estimator="test")


def test_gaussian_knockoff_equi():
    SEED = 42
    n = 100
    p = 50
    X, y, _, non_zero = simu_data(n, p, seed=SEED)
    mu, Sigma = _estimate_distribution(X, shrink=True, cov_estimator="ledoit_wolf")

    X_tilde = gaussian_knockoff_generation(X, mu, Sigma, method="equi", seed=SEED * 2)

    assert X_tilde.shape == (n, p)


def test_gaussian_knockoff_exception():
    SEED = 42
    n = 100
    p = 50
    X, y, _, non_zero = simu_data(n, p, seed=SEED)
    mu, Sigma = _estimate_distribution(X, shrink=True, cov_estimator="ledoit_wolf")

    with pytest.raises(ValueError, match="test is not a valid knockoff contriction method"):
        gaussian_knockoff_generation(X, mu, Sigma, method="test", seed=SEED * 2)

