# -*- coding: utf-8 -*-
# Authors: Binh Nguyen <tuan-binh.nguyen@inria.fr>

from hidimstat.data_simulation import simu_data
from hidimstat.gaussian_knockoff import (
    _estimate_distribution,
    gaussian_knockoff_generation,
)

SEED = 42
fdr = 0.1


def test_estimate_distribution():
    n = 100
    p = 50
    X, y, _, non_zero = simu_data(n, p, seed=SEED)
    mu, Sigma = _estimate_distribution(X, cov_estimator="ledoit_wolf")

    assert mu.size == p
    assert Sigma.shape == (p, p)

    mu, Sigma = _estimate_distribution(X, cov_estimator="graph_lasso")

    assert mu.size == p
    assert Sigma.shape == (p, p)


def test_gaussian_knockoff_equi():
    n = 100
    p = 50
    X, y, _, non_zero = simu_data(n, p, seed=SEED)
    mu, Sigma = _estimate_distribution(X, cov_estimator="ledoit_wolf")

    X_tilde = gaussian_knockoff_generation(X, mu, Sigma, method="equi", seed=SEED * 2)

    assert X_tilde.shape == (n, p)
