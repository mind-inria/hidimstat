"""
Test the dcrt module
"""

import pytest
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.datasets import make_regression, make_classification
from hidimstat.dcrt import d0crt, D0CRT, _lasso_distillation_residual


@pytest.fixture
def generate_regation_dataset(n=100, p=10, noise=0.2, seed=2024):
    X, y = make_regression(n_samples=n, n_features=10, noise=0.2, random_state=2024)
    return X, y


def test_dcrt_lasso_unknow_statistic(generate_regation_dataset):
    """
    Test for unknows statistic
    """
    X, y = generate_regation_dataset
    d0crt = D0CRT(
        screening=False,
        statistic="test",
        random_state=2024,
    )
    # Checking for a different statistic
    with pytest.raises(ValueError, match="test statistic is not supported."):
        d0crt.fit(X, y)


def test_dcrt_lasso_screening(generate_regation_dataset):
    """
    Test for screening parameter and pvalue function
    """
    X, y = generate_regation_dataset
    # Checking with and without screening
    d0crt_no_screening = D0CRT(screening=False, statistic="residual", random_state=2024)
    d0crt_no_screening.fit(X, y)
    sv_no_screening, pvalue_no_screening = d0crt_no_screening.importance()
    d0crt_screening = D0CRT(screening=True, statistic="residual", random_state=2024)
    d0crt_screening.fit(X, y)
    sv_screening, pvalue_screening = d0crt_screening.importance()
    assert np.sum(d0crt_no_screening.selection_features) == 10
    assert np.sum(d0crt_screening.selection_features) < 10
    assert len(sv_no_screening) <= 10
    assert len(pvalue_no_screening) == 10
    assert len(d0crt_no_screening.ts) == 10
    assert len(sv_screening) <= 10
    assert len(pvalue_screening) == 10
    assert len(d0crt_screening.ts) == 10

    # Checking with scaled statistics
    sv_no_screening, pvalue_no_screening = d0crt_no_screening.importance()
    assert len(sv_no_screening) <= 10
    assert len(pvalue_no_screening) == 10
    assert len(d0crt_no_screening.ts) == 10


def test_dcrt_lasso_with_estimed_coefficient(generate_regation_dataset):
    """
    Test the estimated coefficient parameter
    """
    X, y = generate_regation_dataset
    # Checking with random estimated coefficients for the features
    rng = np.random.RandomState(2025)
    estimated_coefs = rng.rand(10)

    d0crt = D0CRT(
        estimated_coef=estimated_coefs,
        screening=False,
        statistic="residual",
        random_state=2026,
    )
    d0crt.fit(X, y)
    sv, pvalue = d0crt.importance()
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.ts) == 10


def test_dcrt_lasso_with_refit(generate_regation_dataset):
    """
    Test the refit parameter
    """
    X, y = generate_regation_dataset
    # Checking with refit
    d0crt_refit = D0CRT(
        refit=True,
        screening=False,
        statistic="residual",
        random_state=2024,
    )
    d0crt_refit.fit(X, y)
    sv, pvalue = d0crt_refit.importance()
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt_refit.ts) == 10


def test_dcrt_lasso_with_no_cv(generate_regation_dataset):
    """
    Test the use_cv parameter
    """
    X, y = generate_regation_dataset
    # Checking with use_cv
    d0crt_use_cv = D0CRT(
        params_lasso_screening={
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
    d0crt_use_cv.fit(X, y)
    sv, pvalue = d0crt_use_cv.importance()
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt_use_cv.ts) == 10


def test_dcrt_lasso_with_covariance(generate_regation_dataset):
    """
    Test dcrt with proviede covariance matrix
    """
    X, y = generate_regation_dataset
    # Checking with a provided covariance matrix
    cov = LedoitWolf().fit(X)

    d0crt_covariance = D0CRT(
        sigma_X=cov.covariance_,
        screening=False,
        statistic="residual",
        random_state=2024,
    )
    d0crt_covariance.fit(X, y)
    sv, pvalue = d0crt_covariance.importance()
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt_covariance.ts) == 10


def test_dcrt_lasso_center():
    """
    Test for unknow statistic
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.2, random_state=2024)
    d0crt = D0CRT(
        centered=False, screening=False, statistic="residual", random_state=2024
    )
    d0crt.fit(X, y)
    sv, pvalue = d0crt.importance(scaled_statistics=True)
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.ts) == 10


def test_dcrt_lasso_refit():
    """
    This function tests the dcrt function using the Lasso learner
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.2, random_state=2024)
    d0crt = D0CRT(refit=True, fit_y=True, statistic="residual", random_state=2024)
    d0crt.fit(X, y)
    sv, pvalue = d0crt.importance(scaled_statistics=True)
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.ts) == 10


def test_dcrt_lasso_no_selection():
    """
    This function tests the dcrt function using the Lasso learner
    with distillation  y using different argument
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.8, random_state=20)
    d0crt = D0CRT(
        estimated_coef=np.ones(10) * 10, statistic="residual", random_state=2024
    )
    d0crt.fit(X, y)
    for arr in [
        d0crt.selection_features,
        d0crt.X_residual,
        d0crt.sigma2,
        d0crt.y_residual,
    ]:
        assert np.all(arr == np.array([]))


def test_dcrt_distillation_x_different():
    """
    This function tests the dcrt function using the Lasso learner
    with distillation x using different argument
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.8, random_state=20)
    d0crt = D0CRT(
        statistic="residual",
        random_state=2024,
        params_lasso_distillation_y={
            "alpha": None,
            "n_alphas": 0,
            "alphas": None,
            "alpha_max_fraction": 0.5,
            "fit_intercept": False,
        },
    )
    d0crt.fit(X, y)
    sv, pvalue = d0crt.importance(scaled_statistics=True)
    assert np.sum(d0crt.selection_features) <= 10
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.ts) == 10


def test_dcrt_distillation_y_different():
    """
    This function tests the dcrt function using the Lasso learner
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.8, random_state=20)
    d0crt = D0CRT(
        statistic="residual",
        random_state=2024,
        params_lasso_distillation_x={
            "alpha": None,
            "n_alphas": 0,
            "alphas": None,
            "alpha_max_fraction": 0.5,
            "fit_intercept": False,
        },
    )
    d0crt.fit(X, y)
    sv, pvalue = d0crt.importance(scaled_statistics=True)
    assert np.sum(d0crt.selection_features) <= 10
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.ts) == 10


def test_dcrt_lasso_fit_with_no_cv():
    """
    Test the dcrt function using the Lasso learner
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.2, random_state=2024)
    d0crt = D0CRT(
        fit_y=True,
        params_lasso_screening={
            "alpha": None,
            "n_alphas": 0,
            "alphas": None,
            "alpha_max_fraction": 0.5,
            "fit_intercept": False,
        },
        screening=False,
        statistic="residual",
        random_state=2026,
    )
    d0crt.fit(X, y)
    sv, pvalue = d0crt.importance(scaled_statistics=True)
    assert np.sum(d0crt.selection_features) <= 10
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.ts) == 10


def test_dcrt_RF_regression():
    """
    This function tests the dcrt function using the Random Forest learner
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.2, random_state=2024)

    d0crt = D0CRT(
        screening=False,
        statistic="random_forest",
        problem_type="regression",
        random_state=2024,
    )
    d0crt.fit(X, y)
    sv, pvalue = d0crt.importance(scaled_statistics=True)
    assert np.sum(d0crt.selection_features) <= 10
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.ts) == 10


def test_dcrt_RF_classification():
    """
    This function tests the dcrt function using the Random Forest learner
    """
    X, y = make_classification(n_samples=100, n_features=10, random_state=2024)
    d0crt = D0CRT(
        screening=False,
        statistic="random_forest",
        problem_type="classification",
        random_state=2024,
    )
    d0crt.fit(X, y)
    sv, pvalue = d0crt.importance(scaled_statistics=True)
    assert np.sum(d0crt.selection_features) <= 10
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.ts) == 10


def test_exception_not_fitted():
    """Test if an exception is raise when the methosd is not fitted"""
    X, y = make_classification(n_samples=100, n_features=10, random_state=2024)
    d0crt = D0CRT(
        screening=False,
        statistic="random_forest",
        problem_type="classification",
        random_state=2024,
    )
    with pytest.raises(
        ValueError, match="The D0CRT requires to be fit before any analysis"
    ):
        _, _ = d0crt.importance(scaled_statistics=True)


def test_exception_lasso_distillation_residual():
    """
    This function tests the dcrt function using the Lasso learner
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.2, random_state=2024)
    with pytest.raises(
        ValueError, match="Either fit_y is true or coeff_full must be provided."
    ):
        _lasso_distillation_residual(X, y, 0)


def test_function_d0crt():
    """Test the function dcrt"""
    X, y = make_regression(n_samples=100, n_features=10, noise=0.2, random_state=2024)
    sv, pvalue = d0crt(X, y)
    assert len(sv) <= 10
    assert len(pvalue) == 10
