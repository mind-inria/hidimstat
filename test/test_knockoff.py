from hidimstat.knockoffs import (
    model_x_knockoff_aggregation,
    model_x_knockoff,
    model_x_knockoff_filter,
    model_x_knockoff_pvalue,
    model_x_knockoff_bootstrap_e_value,
    model_x_knockoff_bootstrap_quantile,
)
from hidimstat.gaussian_knockoff import gaussian_knockoff_generation, _s_equi
from hidimstat.data_simulation import simu_data
from hidimstat.utils import cal_fdp_power
import numpy as np
import pytest
from sklearn.covariance import LedoitWolf, GraphicalLassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold


def test_knockoff_aggregation():
    n = 500
    p = 100
    snr = 5
    n_bootstraps = 25
    fdr = 0.5
    X, y, _, non_zero_index = simu_data(n, p, snr=snr, seed=0)

    test_scores = model_x_knockoff_aggregation(
        X, y, n_bootstraps=n_bootstraps, random_state=0
    )
    selected_verbose, aggregated_pval, pvals = model_x_knockoff_bootstrap_quantile(
        test_scores, fdr=fdr, selection_only=False
    )

    fdp_verbose, power_verbose = cal_fdp_power(selected_verbose, non_zero_index)

    selected_no_verbose = model_x_knockoff_bootstrap_quantile(
        test_scores, fdr=fdr, selection_only=True
    )

    fdp_no_verbose, power_no_verbose = cal_fdp_power(
        selected_no_verbose, non_zero_index
    )

    assert pvals.shape == (n_bootstraps, p)
    assert fdp_verbose < 0.5
    assert power_verbose > 0.1
    assert fdp_no_verbose < 0.5
    assert power_no_verbose > 0.1
    np.testing.assert_array_equal(selected_no_verbose, selected_verbose)

    # Single AKO (or vanilla KO) (verbose vs no verbose)
    test_bootstrap = model_x_knockoff_aggregation(X, y, n_bootstraps=2, random_state=5)
    test_no_bootstrap = model_x_knockoff(
        X, y, seed=np.random.RandomState(5).randint(1, np.iinfo(np.int32).max, 2)[0]
    )

    np.testing.assert_array_equal(test_bootstrap[0], test_no_bootstrap)

    # Using e-values aggregation (verbose vs no verbose)

    selected_verbose, aggregated_eval, evals = model_x_knockoff_bootstrap_e_value(
        test_scores, fdr=fdr, selection_only=False
    )
    fdp_verbose, power_verbose = cal_fdp_power(selected_verbose, non_zero_index)
    selected_no_verbose = model_x_knockoff_bootstrap_e_value(
        test_scores, fdr=fdr, selection_only=True
    )
    fdp_no_verbose, power_no_verbose = cal_fdp_power(
        selected_no_verbose, non_zero_index
    )

    assert pvals.shape == (n_bootstraps, p)
    assert fdp_verbose < 0.5
    assert power_verbose > 0.1
    assert fdp_no_verbose < 0.5
    assert power_no_verbose > 0.1

    # Checking wrong type for random_state
    with pytest.raises(Exception):
        _ = model_x_knockoff_aggregation(
            X,
            y,
            random_state="test",
        )

    # Checking value for offset not belonging to (0,1)
    with pytest.raises(Exception):
        _ = model_x_knockoff_bootstrap_quantile(
            test_scores,
            offset=2,
        )

    with pytest.raises(Exception):
        _ = model_x_knockoff_bootstrap_e_value(
            test_scores,
            offset=2,
        )


def test_model_x_knockoff():
    seed = 42
    fdr = 0.2
    n = 300
    p = 300
    X, y, _, non_zero = simu_data(n, p, seed=seed)
    test_score = model_x_knockoff(X, y, seed=seed + 1)
    ko_result = model_x_knockoff_filter(
        test_score,
        fdr=fdr,
    )
    fdp, power = cal_fdp_power(ko_result, non_zero)
    assert fdp <= 0.2
    assert power > 0.7

    ko_result = model_x_knockoff_pvalue(test_score, fdr=fdr, selection_only=True)
    fdp, power = cal_fdp_power(ko_result, non_zero)
    assert fdp <= 0.2
    assert power > 0.7


def test_model_x_knockoff_estimator():
    seed = 42
    fdr = 0.2
    n = 300
    p = 300
    X, y, _, non_zero = simu_data(n, p, seed=seed)
    test_score = model_x_knockoff(
        X,
        y,
        estimator=GridSearchCV(Lasso(), param_grid={"alpha": np.linspace(0.2, 0.3, 5)}),
        preconfigure_estimator=None,
        seed=seed + 1,
    )
    ko_result = model_x_knockoff_filter(
        test_score,
        fdr=fdr,
    )
    fdp, power = cal_fdp_power(ko_result, non_zero)

    assert fdp <= 0.2
    assert power > 0.7


def test_estimate_distribution():
    """
    test different estimation of the covariance
    """
    seed = 42
    fdr = 0.1
    n = 100
    p = 50
    X, y, _, non_zero = simu_data(n, p, seed=seed)
    test_score = model_x_knockoff(X, y, cov_estimator=LedoitWolf(assume_centered=True),
                                  seed=seed+1)
    ko_result = model_x_knockoff_filter(
        test_score,
        fdr=fdr,
    )
    for i in ko_result:
        assert np.any(i == non_zero)
    test_score = model_x_knockoff(
        X,
        y,
        cov_estimator=GraphicalLassoCV(
            alphas=[1e-3, 1e-2, 1e-1, 1],
            cv=KFold(n_splits=5, shuffle=True, random_state=0),
        ),
        seed=seed+2
    )
    ko_result = model_x_knockoff_filter(
        test_score,
        fdr=fdr,
    )
    for i in ko_result:
        assert np.any(i == non_zero)


def test_gaussian_knockoff_equi():
    seed = 42
    n = 100
    p = 50
    X, y, _, non_zero = simu_data(n, p, seed=seed)
    mu = X.mean(axis=0)
    Sigma = LedoitWolf(assume_centered=True).fit(X).covariance_

    X_tilde = gaussian_knockoff_generation(X, mu, Sigma, seed=seed * 2)

    assert X_tilde.shape == (n, p)


def test_s_equi_not_define_positive():
    """test the warning and error of s_equi function"""
    n= 10 
    tol = 1e-7
    np.random.seed(42)

    # random matrix
    sigma = np.random.randn(n,n)
    sigma -= np.min(sigma)
    with pytest.raises(Exception, match='The covariance matrix is not positive-definite.'):
        _s_equi(sigma)

    # positive matrix
    while not np.all(np.linalg.eigvalsh(sigma) > tol):
        sigma += 0.1 * np.eye(n)
        print(np.linalg.eigvalsh(sigma))
    with pytest.warns(UserWarning, match='The equi-correlated matrix'):
        _s_equi(sigma)

    # positive definite matrix
    sigma = sigma.T*sigma
    sigma = (sigma + sigma.T)/2
    _s_equi(sigma)