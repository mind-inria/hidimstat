# from hidimstat.knockoffs import ModelXKnockoff

from hidimstat.conditional_randomization_test import (
    CRT as ModelXKnockoff,
)
from hidimstat.statistical_tools.gaussian_knockoff import GaussianGenerator
from hidimstat.statistical_tools.lasso_test import lasso_statistic_with_sampling
from hidimstat._utils.scenario import multivariate_simulation
from hidimstat.statistical_tools.multiple_testing import fdp_power
import numpy as np
import pytest
from sklearn.covariance import LedoitWolf, GraphicalLassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor


def test_knockoff_bootstrap_quantile():
    """Test bootstrap knockoof with quantile aggregation"""
    n = 500
    p = 100
    signal_noise_ratio = 5
    n_sampling = 25
    fdr = 0.5
    X, y, beta, noise = multivariate_simulation(
        n, p, signal_noise_ratio=signal_noise_ratio, seed=0
    )
    non_zero_index = np.where(beta)[0]

    model_x_knockoff = ModelXKnockoff(n_sampling=n_sampling)
    model_x_knockoff.fit(X).importance(X, y)
    selected = model_x_knockoff.selection_fdr(fdr=fdr)

    fdp, power = fdp_power(np.where(selected)[0], non_zero_index)

    assert model_x_knockoff.test_scores_.shape == (n_sampling, p)
    assert fdp < 0.5
    assert power > 0.1


def test_knockoff_bootstrap_e_values():
    """Test bootstrap Knockoff with e-values"""
    n = 500
    p = 100
    signal_noise_ratio = 5
    n_sampling = 25
    fdr = 0.5
    X, y, beta, noise = multivariate_simulation(
        n, p, signal_noise_ratio=signal_noise_ratio, seed=0
    )
    non_zero_index = np.where(beta)[0]

    model_x_knockoff = ModelXKnockoff(n_sampling=n_sampling)
    model_x_knockoff.fit(X).importance(X, y)
    selected = model_x_knockoff.selection_fdr(
        fdr=fdr / 2, evalues=True, fdr_control="ebh"
    )

    # Using e-values aggregation (verbose vs no verbose)
    fdp, power = fdp_power(np.where(selected)[0], non_zero_index)

    assert model_x_knockoff.test_scores_.shape == (n_sampling, p)
    assert fdp < 0.5
    assert power > 0.1


def test_invariant_with_bootstrap():
    """Test bootstrap Knockoff"""
    n = 500
    p = 100
    signal_noise_ratio = 5
    fdr = 0.5
    X, y, beta, noise = multivariate_simulation(
        n, p, signal_noise_ratio=signal_noise_ratio, seed=0
    )
    # Single AKO (or vanilla KO) (verbose vs no verbose)
    model_x_knockoff_bootstrap = ModelXKnockoff(
        n_sampling=5,
        generator=GaussianGenerator(
            cov_estimator=LedoitWolf(assume_centered=True), random_state=5
        ),
    )
    model_x_knockoff_bootstrap.fit(X).importance(X, y)
    selected_bootstrap = model_x_knockoff_bootstrap.selection_fdr(fdr=fdr)
    model_x_knockoff_no_bootstrap = ModelXKnockoff(
        n_sampling=1,
        generator=GaussianGenerator(
            cov_estimator=LedoitWolf(assume_centered=True), random_state=5
        ),
    )
    model_x_knockoff_no_bootstrap.fit(X).importance(X, y)
    selected_no_bootstrap = model_x_knockoff_no_bootstrap.selection_fdr(fdr=fdr)

    np.testing.assert_array_equal(
        model_x_knockoff_bootstrap.test_scores_[0],
        model_x_knockoff_no_bootstrap.test_scores_[0],
    )
    np.testing.assert_array_equal(selected_bootstrap, selected_no_bootstrap)


def test_model_x_knockoff():
    """Test the selection of variable from knockoff"""
    seed = 42
    fdr = 0.2
    n = 300
    p = 300
    support_size = 18
    X, y, beta, noise = multivariate_simulation(
        n, p, support_size=support_size, seed=seed
    )
    non_zero = np.where(beta)[0]
    model_x_knockoff = ModelXKnockoff(n_sampling=5)
    model_x_knockoff.fit(X).importance(X, y)
    selected = model_x_knockoff.selection_fdr(fdr=fdr)

    fdp, power = fdp_power(np.where(selected)[0], non_zero)
    assert fdp <= 0.2
    assert power > 0.7
    assert np.all(0 <= model_x_knockoff.pvalues_) or np.all(
        model_x_knockoff.pvalues_ <= 1
    )


def test_model_x_knockoff_estimator():
    """Test knockoff with a crossvalidation estimator"""
    seed = 42
    fdr = 0.2
    n = 300
    p = 300
    X, y, beta, noise = multivariate_simulation(n, p, seed=seed)
    non_zero = np.where(beta)[0]

    def lasso_statistic(X, X_tilde, y):
        return lasso_statistic_with_sampling(
            X,
            X_tilde,
            y,
            lasso=GridSearchCV(Lasso(), param_grid={"alpha": np.linspace(0.2, 0.3, 5)}),
            preconfigure_lasso=None,
        )

    model_x_knockoff = ModelXKnockoff(
        n_sampling=1,
        generator=GaussianGenerator(
            cov_estimator=LedoitWolf(assume_centered=True), random_state=seed
        ),
        statistical_test=lasso_statistic,
    )
    model_x_knockoff.fit(X).importance(X, y)
    selected = model_x_knockoff.selection_fdr(fdr=fdr)

    fdp, power = fdp_power(np.where(selected)[0], non_zero)

    assert fdp <= fdr
    assert power > 0.7


# def test_model_x_knockoff_exception():
#     "Test the exception raise by model_x_knockoff"
#     n = 50
#     p = 100
#     seed = 45
#     rgn = np.random.RandomState(seed)
#     X = rgn.randn(n, p)
#     y = rgn.randn(n)
#     with pytest.raises(TypeError, match="You should not use this function"):
#         model_x_knockoff(
#             X,
#             y,
#             estimator=Lasso(),
#             n_sampling=1,
#         )
#     with pytest.raises(TypeError, match="estimator should be linear"):
#         model_x_knockoff(
#             X,
#             y,
#             estimator=DecisionTreeRegressor(),
#             preconfigure_estimator=None,
#             n_sampling=1,
#         )


def test_estimate_distribution():
    """
    test different estimation of the covariance
    """
    seed = 42
    fdr = 0.1
    n = 100
    p = 50
    X, y, beta, noise = multivariate_simulation(n, p, seed=seed)
    non_zero = np.where(beta)[0]
    model_x_knockoff = ModelXKnockoff(
        n_sampling=1,
        generator=GaussianGenerator(
            cov_estimator=LedoitWolf(assume_centered=True), random_state=seed + 1
        ),
    )
    model_x_knockoff.fit(X).importance(X, y)
    selected = model_x_knockoff.selection_fdr(fdr=fdr)
    for i in np.where(selected)[0]:
        assert np.any(i == non_zero)

    model_x_knockoff = ModelXKnockoff(
        n_sampling=1,
        generator=GaussianGenerator(
            cov_estimator=GraphicalLassoCV(
                alphas=[1e-3, 1e-2, 1e-1, 1],
                cv=KFold(n_splits=5, shuffle=True, random_state=0),
            ),
            random_state=seed + 2,
        ),
    )
    model_x_knockoff.fit(X).importance(X, y)
    selected = model_x_knockoff.selection_fdr(fdr=fdr)
    for i in np.where(selected)[0]:
        assert np.any(i == non_zero)
