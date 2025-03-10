# -*- coding: utf-8 -*-
# Authors: Binh Nguyen <tuan-binh.nguyen@inria.fr>

from hidimstat.data_simulation import simu_data
from hidimstat.gaussian_knockoff import (
    _estimate_distribution,
    gaussian_knockoff_generation,
)
import pytest


def test_estimate_distribution_ledoit_wolf():
    seed = 42
    n = 100
    p = 50
    X, _, _, _ = simu_data(n, p, seed=seed)
    mu, sigma = _estimate_distribution(X, cov_estimator="ledoit_wolf")

    assert mu.size == p
    assert sigma.shape == (p, p)


def test_estimate_distribution_lasso():
    seed = 42
    n = 100
    p = 50
    X, _, _, _ = simu_data(n, p, seed=seed)
    mu, sigma = _estimate_distribution(X, shrink=True, cov_estimator="graph_lasso")

    assert mu.size == p
    assert sigma.shape == (p, p)


def test_gaussian_knockoff_estimate_exception():
    seed = 42
    n = 100
    p = 50
    X, _, _, _ = simu_data(n, p, seed=seed)
    with pytest.raises(
        ValueError, match="test is not a valid covariance estimated method"
    ):
        _estimate_distribution(X, shrink=True, cov_estimator="test")


def test_gaussian_knockoff_equi():
    seed = 42
    n = 100
    p = 50
    X, _, _, _ = simu_data(n, p, seed=seed)
    mu, sigma = _estimate_distribution(X, shrink=True, cov_estimator="ledoit_wolf")

    X_tilde = gaussian_knockoff_generation(X, mu, sigma, method="equi", seed=seed * 2)

    assert X_tilde.shape == (n, p)


def test_gaussian_knockoff_exception():
    seed = 42
    n = 100
    p = 50
    X, _, _, _ = simu_data(n, p, seed=seed)
    mu, sigma = _estimate_distribution(X, shrink=True, cov_estimator="ledoit_wolf")

    with pytest.raises(
        ValueError, match="test is not a valid knockoff contriction method"
    ):
        gaussian_knockoff_generation(X, mu, sigma, method="test", seed=seed * 2)
