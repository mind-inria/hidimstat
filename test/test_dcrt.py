"""
Test the dcrt module
"""

import pytest
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.datasets import make_regression, make_classification

from hidimstat.dcrt import dcrt_zero, dcrt_pvalue, _lasso_distillation_residual


@pytest.fixture
def generate_regation_dataset(n=100, p=10, noise=0.2, seed=2024):
    X, y = make_regression(n_samples=n, n_features=10, noise=0.2, random_state=2024)
    return X, y


def test_dcrt_lasso_unknow_statistic(generate_regation_dataset):
    """
    Test for unknows statistic
    """
    X, y = generate_regation_dataset
    # Checking for a different statistic
    with pytest.raises(ValueError, match="test statistic is not supported."):
        _ = dcrt_zero(
            X,
            y,
            screening=False,
            statistic="test",
            random_state=2024,
        )


def test_dcrt_lasso_screening(generate_regation_dataset):
    """
    Test for screening parameter and pvalue function
    """
    X, y = generate_regation_dataset
    # Checking with and without screening
    results_no_screening = dcrt_zero(
        X, y, screening=False, statistic="residual", random_state=2024
    )
    vi_no_screening = dcrt_pvalue(
        results_no_screening[0],
        results_no_screening[1],
        results_no_screening[2],
        results_no_screening[3],
    )
    results_screening = dcrt_zero(
        X, y, screening=True, statistic="residual", random_state=2024
    )
    vi_screening = dcrt_pvalue(
        results_screening[0],
        results_screening[1],
        results_screening[2],
        results_screening[3],
    )
    assert np.sum(results_no_screening[0]) == 10
    assert np.sum(results_screening[0]) < 10
    assert len(vi_no_screening[0]) <= 10
    assert len(vi_no_screening[1]) == 10
    assert len(vi_no_screening[2]) == 10
    assert len(vi_screening[0]) <= 10
    assert len(vi_screening[1]) == 10
    assert len(vi_screening[2]) == 10

    # Checking with scaled statistics
    vi_scaled = dcrt_pvalue(
        results_no_screening[0],
        results_no_screening[1],
        results_no_screening[2],
        results_no_screening[3],
    )
    assert len(vi_scaled[0]) <= 10
    assert len(vi_scaled[1]) == 10
    assert len(vi_scaled[2]) == 10


def test_dcrt_lasso_with_estimed_coefficient(generate_regation_dataset):
    """
    Test the estimated coefficient parameter
    """
    X, y = generate_regation_dataset
    # Checking with random estimated coefficients for the features
    rng = np.random.RandomState(2025)
    estimated_coefs = rng.rand(10)

    results = dcrt_zero(
        X,
        y,
        estimated_coef=estimated_coefs,
        screening=False,
        statistic="residual",
        random_state=2026,
    )
    vi = dcrt_pvalue(results[0], results[1], results[2], results[3])
    assert len(vi[0]) <= 10
    assert len(vi[1]) == 10
    assert len(vi[2]) == 10


def test_dcrt_lasso_with_refit(generate_regation_dataset):
    """
    Test the refit parameter
    """
    X, y = generate_regation_dataset
    # Checking with refit
    results_refit = dcrt_zero(
        X,
        y,
        refit=True,
        screening=False,
        statistic="residual",
        random_state=2024,
    )
    vi_refit = dcrt_pvalue(
        results_refit[0],
        results_refit[1],
        results_refit[2],
        results_refit[3],
    )
    assert len(vi_refit[0]) <= 10
    assert len(vi_refit[1]) == 10
    assert len(vi_refit[2]) == 10


def test_dcrt_lasso_with_no_cv(generate_regation_dataset):
    """
    Test the use_cv parameter
    """
    X, y = generate_regation_dataset
    # Checking with use_cv
    results_use_cv = dcrt_zero(
        X,
        y,
        kargs_lasso_estimator={
            "alpha": None,
            "n_alphas": 0,
            "alphas": None,
            "alpha_max_fraction": 0.5,
            "fit_intercept": False,
        },
        screening=False,
        statistic="residual",
        random_state=2024,
    )
    vi_use_cv = dcrt_pvalue(
        results_use_cv[0],
        results_use_cv[1],
        results_use_cv[2],
        results_use_cv[3],
    )
    assert len(vi_use_cv[0]) <= 10
    assert len(vi_use_cv[1]) == 10
    assert len(vi_use_cv[1]) == 10


def test_dcrt_lasso_with_covariance(generate_regation_dataset):
    """
    Test dcrt with proviede covariance matrix
    """
    X, y = generate_regation_dataset
    # Checking with a provided covariance matrix
    cov = LedoitWolf().fit(X)

    results_covariance = dcrt_zero(
        X,
        y,
        sigma_X=cov.covariance_,
        screening=False,
        statistic="residual",
        random_state=2024,
    )
    vi_covariance = dcrt_pvalue(
        results_covariance[0],
        results_covariance[1],
        results_covariance[2],
        results_covariance[3],
    )
    assert len(vi_covariance[0]) <= 10
    assert len(vi_covariance[1]) == 10
    assert len(vi_covariance[1]) == 10


def test_dcrt_lasso_center():
    """
    This function tests the dcrt function using the Lasso learner
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.2, random_state=2024)
    results = dcrt_zero(
        X, y, centered=False, screening=False, statistic="residual", random_state=2024
    )
    vi = dcrt_pvalue(
        results[0],
        results[1],
        results[2],
        results[3],
    )
    assert np.sum(results[0]) == 10
    assert len(vi[0]) <= 10
    assert len(vi[1]) == 10
    assert len(vi[2]) == 10
    vi = dcrt_pvalue(
        results[0],
        results[1],
        results[2],
        results[3],
        scaled_statistics=True,
    )
    assert len(vi[0]) <= 10
    assert len(vi[1]) == 10
    assert len(vi[2]) == 10


def test_dcrt_lasso_refit():
    """
    This function tests the dcrt function using the Lasso learner
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.2, random_state=2024)
    results = dcrt_zero(
        X, y, refit=True, fit_y=True, statistic="residual", random_state=2024
    )
    vi = dcrt_pvalue(
        results[0],
        results[1],
        results[2],
        results[3],
    )
    assert np.sum(results[0]) <= 10
    assert len(vi[0]) <= 10
    assert len(vi[1]) == 10
    assert len(vi[2]) == 10


def test_dcrt_lasso_no_selection():
    """
    This function tests the dcrt function using the Lasso learner
    with distillation  y using different argument
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.8, random_state=20)
    results = dcrt_zero(
        X, y, estimated_coef=np.ones(10) * 10, statistic="residual", random_state=2024
    )
    for i in results:
        assert np.all(i == np.array([]))


def test_dcrt_distillation_x_different():
    """
    This function tests the dcrt function using the Lasso learner
    with distillation x using different argument
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.8, random_state=20)
    results = dcrt_zero(
        X,
        y,
        statistic="residual",
        random_state=2024,
        kargs_lasso_distillation_y={
            "alpha": None,
            "n_alphas": 0,
            "alphas": None,
            "alpha_max_fraction": 0.5,
            "fit_intercept": False,
        },
    )
    vi = dcrt_pvalue(
        results[0],
        results[1],
        results[2],
        results[3],
    )
    assert np.sum(results[0]) <= 10
    assert len(vi[0]) <= 10
    assert len(vi[1]) == 10
    assert len(vi[2]) == 10


def test_dcrt_distillation_y_different():
    """
    This function tests the dcrt function using the Lasso learner
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.8, random_state=20)
    results = dcrt_zero(
        X,
        y,
        statistic="residual",
        random_state=2024,
        kargs_lasso_distillation_x={
            "alpha": None,
            "n_alphas": 0,
            "alphas": None,
            "alpha_max_fraction": 0.5,
            "fit_intercept": False,
        },
    )
    vi = dcrt_pvalue(
        results[0],
        results[1],
        results[2],
        results[3],
    )
    assert np.sum(results[0]) <= 10
    assert len(vi[0]) <= 10
    assert len(vi[1]) == 10
    assert len(vi[2]) == 10


def test_dcrt_lasso_fit_with_no_cv():
    """
    Test the dcrt function using the Lasso learner
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.2, random_state=2024)
    results = dcrt_zero(
        X,
        y,
        fit_y=True,
        kargs_lasso_estimator={
            "alpha": None,
            "n_alphas": 0,
            "alphas": None,
            "alpha_max_fraction": 0.5,
            "fit_intercept": False,
        },
        screening=False,
        statistic="residual",
        random_state=2024,
    )
    vi = dcrt_pvalue(
        results[0],
        results[1],
        results[2],
        results[3],
    )
    assert np.sum(results[0]) == 10
    assert len(vi[0]) <= 10
    assert len(vi[1]) == 10
    assert len(vi[2]) == 10


def test_dcrt_RF_regression():
    """
    This function tests the dcrt function using the Random Forest learner
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.2, random_state=2024)

    with pytest.raises(Exception):
        _ = dcrt_zero(
            X,
            y,
            screening=False,
            loss="test",
            random_state=2024,
            statistic="randomforest",
        )

    res = dcrt_zero(
        X,
        y,
        screening=False,
        statistic="randomforest",
        problem_type="regression",
        random_state=2024,
    )
    vi = dcrt_pvalue(res[0], res[1], res[2], res[3])
    assert len(vi[0]) <= 10
    assert len(vi[1]) == 10
    assert len(vi[2]) == 10


def test_dcrt_RF_classification():
    """
    This function tests the dcrt function using the Random Forest learner
    """
    X, y = make_classification(n_samples=100, n_features=10, random_state=2024)
    res = dcrt_zero(
        X,
        y,
        screening=False,
        statistic="randomforest",
        problem_type="classification",
        random_state=2024,
    )
    vi = dcrt_pvalue(res[0], res[1], res[2], res[3])
    assert len(vi[0]) <= 10
    assert len(vi[1]) == 10
    assert len(vi[2]) == 10


def test_exception_lasso_distillation_residual():
    """
    This function tests the dcrt function using the Lasso learner
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.2, random_state=2024)
    with pytest.raises(
        ValueError, match="Either fit_y is true or coeff_full must be provided."
    ):
        _lasso_distillation_residual(X, y, 0)
