"""
Test the dcrt module
"""

import pytest
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import KFold
from hidimstat.dcrt import d0crt, D0CRT
from sklearn.linear_model import LassoCV, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


from hidimstat._utils.regression import _alpha_max


@pytest.fixture
def generate_regation_dataset(n=100, p=10, noise=0.2, seed=2024):
    X, y = make_regression(n_samples=n, n_features=10, noise=0.2, random_state=2024)
    return X, y


def test_dcrt_lasso_screening(generate_regation_dataset):
    """
    Test for screening parameter and pvalue function
    """
    X, y = generate_regation_dataset
    # Checking with and without screening
    d0crt_no_screening = D0CRT(
        estimator=LassoCV(n_jobs=1), screening=False, random_state=2024
    )
    pvalue_no_screening = d0crt_no_screening.fit_importance(X, y)
    sv_no_screening = d0crt_no_screening.selection(threshold_pvalue=0.05)
    d0crt_screening = D0CRT(
        estimator=LassoCV(n_jobs=1), screening=True, random_state=2024
    )
    pvalue_screening = d0crt_screening.fit_importance(X, y)
    sv_screening = d0crt_screening.selection(threshold_pvalue=0.05)
    assert np.sum(d0crt_no_screening.ts_ != 0) <= 10
    assert np.sum(d0crt_screening.ts_ != 0) <= 10
    assert len(sv_no_screening) <= 10
    assert len(pvalue_no_screening) == 10
    assert len(d0crt_no_screening.ts_) == 10
    assert len(sv_screening) <= 10
    assert len(pvalue_screening) == 10
    assert len(d0crt_screening.ts_) == 10

    # Checking with scaled statistics
    d0crt_no_screening = D0CRT(
        estimator=LassoCV(n_jobs=1),
        screening=False,
        random_state=2024,
        scaled_statistics=True,
    )
    d0crt_no_screening.fit_importance(X, y)
    pvalue_no_screening = d0crt_no_screening.importance(X, y)
    sv_no_screening = d0crt_no_screening.selection(threshold_pvalue=0.05)
    assert len(sv_no_screening) <= 10
    assert len(pvalue_no_screening) == 10
    assert len(d0crt_no_screening.ts_) == 10


def test_dcrt_lasso_with_estimed_coefficient(generate_regation_dataset):
    """
    Test the estimated coefficient parameter
    """
    X, y = generate_regation_dataset
    # Checking with random estimated coefficients for the features
    rng = np.random.RandomState(2025)
    estimated_coefs = rng.rand(10)

    d0crt = D0CRT(
        estimator=LassoCV(n_jobs=1),
        estimated_coef=estimated_coefs,
        screening=False,
        random_state=2026,
    )
    d0crt.fit(X, y)
    pvalue = d0crt.importance(X, y)
    sv = d0crt.selection(threshold_pvalue=0.05)
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.ts_) == 10


def test_dcrt_lasso_with_refit(generate_regation_dataset):
    """
    Test the refit parameter
    """
    X, y = generate_regation_dataset
    # Checking with refit
    d0crt_refit = D0CRT(
        estimator=LassoCV(n_jobs=1),
        refit=True,
        screening=False,
        random_state=2024,
    )
    pvalue = d0crt_refit.fit_importance(X, y)
    sv = d0crt_refit.selection(threshold_pvalue=0.05)
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt_refit.ts_) == 10


def test_dcrt_lasso_with_no_cv(generate_regation_dataset):
    """
    Test the parameters to the Lasso of x-distillation
    """
    X, y = generate_regation_dataset
    # Checking with use_cv
    d0crt_use_cv = D0CRT(
        estimator=LassoCV(n_jobs=1),
        params_lasso_screening={
            "alpha": None,
            "n_alphas": 0,
            "alphas": None,
            "alpha_max_fraction": 0.5,
            "fit_intercept": False,
        },
        screening=False,
        random_state=2024,
    )
    pvalue = d0crt_use_cv.fit_importance(X, y)
    sv = d0crt_use_cv.selection(threshold_pvalue=0.05)
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt_use_cv.ts_) == 10


def test_dcrt_lasso_with_covariance(generate_regation_dataset):
    """
    Test dcrt with proviede covariance matrix
    """
    X, y = generate_regation_dataset
    # Checking with a provided covariance matrix
    cov = LedoitWolf().fit(X)

    d0crt_covariance = D0CRT(
        estimator=LassoCV(n_jobs=1),
        sigma_X=cov.covariance_,
        screening=False,
        random_state=2024,
    )
    pvalue = d0crt_covariance.fit_importance(X, y)
    sv = d0crt_covariance.selection(threshold_pvalue=0.05)
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt_covariance.ts_) == 10


def test_dcrt_lasso_center():
    """
    Test for not center the data
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.2, random_state=2024)
    d0crt = D0CRT(
        estimator=LassoCV(n_jobs=1),
        centered=False,
        screening=False,
        random_state=2024,
        scaled_statistics=True,
    )
    pvalue = d0crt.fit_importance(X, y)
    sv = d0crt.selection(threshold_pvalue=0.05)
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.ts_) == 10


def test_dcrt_lasso_refit():
    """
    This function tests the dcrt function using the Lasso learner and refit
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.2, random_state=2024)
    d0crt = D0CRT(
        estimator=LassoCV(n_jobs=1),
        refit=True,
        fit_y=True,
        random_state=2024,
        scaled_statistics=True,
    )
    pvalue = d0crt.fit_importance(X, y)
    sv = d0crt.selection(threshold_pvalue=0.05)
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.ts_) == 10


def test_dcrt_lasso_no_selection():
    """
    This function tests the dcrt function using the Lasso learner
    with distillation y using different argument
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.8, random_state=20)
    d0crt = D0CRT(
        estimator=LassoCV(n_jobs=1), estimated_coef=np.ones(10) * 10, random_state=2024
    )
    d0crt.fit(X, y)
    for arr in [
        d0crt.clf_x_,
        d0crt.clf_y_,
    ]:
        assert np.all(arr == np.array([]))


def test_dcrt_distillation_x_different():
    """
    This function tests the dcrt function using the Lasso learner
    with distillation x using different argument
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.8, random_state=20)
    d0crt = D0CRT(
        estimator=Lasso(alpha=0.5 * _alpha_max(X, y), fit_intercept=False),
        random_state=2024,
        scaled_statistics=True,
    )
    pvalue = d0crt.fit_importance(X, y)
    sv = d0crt.selection(threshold_pvalue=0.05)
    assert np.where(d0crt.ts_ != 0)[0].shape[0] <= 10
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.ts_) == 10


def test_dcrt_distillation_y_different():
    """
    This function tests the dcrt function using the Lasso learner
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.8, random_state=20)
    d0crt = D0CRT(
        estimator=LassoCV(n_jobs=1),
        random_state=2024,
        params_lasso_distillation_x={
            "alpha": None,
            "n_alphas": 0,
            "alphas": None,
            "alpha_max_fraction": 0.5,
            "fit_intercept": False,
        },
        scaled_statistics=True,
    )
    pvalue = d0crt.fit_importance(X, y)
    sv = d0crt.selection(threshold_pvalue=0.05)
    assert np.where(d0crt.ts_ != 0)[0].shape[0] <= 10
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.ts_) == 10


def test_dcrt_lasso_fit_with_no_cv():
    """
    Test the dcrt function using the Lasso learner
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.2, random_state=2024)
    d0crt = D0CRT(
        estimator=LassoCV(n_jobs=1),
        fit_y=True,
        params_lasso_screening={
            "alpha": None,
            "n_alphas": 0,
            "alphas": None,
            "alpha_max_fraction": 0.5,
            "fit_intercept": False,
        },
        screening=False,
        random_state=2026,
        scaled_statistics=True,
    )
    pvalue = d0crt.fit_importance(X, y)
    sv = d0crt.selection(threshold_pvalue=0.05)
    assert np.sum(d0crt.ts_ != 0) <= 10
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.ts_) == 10


def test_dcrt_RF_regression():
    """
    This function tests the dcrt function using the Random Forest learner
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.2, random_state=2024)

    d0crt = D0CRT(
        estimator=RandomForestRegressor(n_estimators=100, random_state=2026, n_jobs=1),
        method="predict",
        screening=False,
        random_state=2026,
        scaled_statistics=True,
    )
    pvalue = d0crt.fit_importance(X, y)
    sv = d0crt.selection(threshold_pvalue=0.05)
    assert np.where(d0crt.ts_ != 0)[0].shape[0] <= 10
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.ts_) == 10


def test_dcrt_RF_classification():
    """
    This function tests the dcrt function using the Random Forest learner
    """
    X, y = make_classification(n_samples=100, n_features=10, random_state=2024)
    d0crt = D0CRT(
        estimator=RandomForestClassifier(n_estimators=100, random_state=2026, n_jobs=1),
        method="predict_proba",
        screening=False,
        random_state=2026,
        scaled_statistics=True,
    )
    pvalue = d0crt.fit_importance(X, y)
    sv = d0crt.selection(threshold_pvalue=0.05)
    assert np.where(d0crt.ts_ != 0)[0].shape[0] <= 10
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.ts_) == 10


def test_exception_not_fitted():
    """Test if an exception is raised when the method is not fitted"""
    X, y = make_classification(n_samples=100, n_features=10, random_state=2024)
    d0crt = D0CRT(
        estimator=RandomForestClassifier(n_estimators=100, random_state=2026, n_jobs=1),
        method="predict_proba",
        screening=False,
        random_state=2024,
        scaled_statistics=True,
    )
    with pytest.raises(
        ValueError, match="The D0CRT requires to be fit before any analysis"
    ):
        _, _ = d0crt.importance(X, y)


def test_warning_not_used_parameters():
    """Test if an exception is raised when the method is not fitted"""
    X, y = make_classification(n_samples=100, n_features=10, random_state=2024)
    d0crt = D0CRT(
        estimator=RandomForestClassifier(n_estimators=100, random_state=2026, n_jobs=1),
        method="predict_proba",
        screening=False,
        random_state=2024,
    )
    d0crt.fit(X, y)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    with pytest.warns(UserWarning, match="cv won't be used"):
        _ = d0crt.fit_importance(X, y, cv=cv)


def test_function_d0crt():
    """Test the d0crt function"""
    X, y = make_regression(n_samples=100, n_features=10, noise=0.2, random_state=2024)
    sv, importances, pvalues = d0crt(LassoCV(n_jobs=1), X, y)
    assert len(sv) <= 10
    assert len(importances) == 10
    assert len(pvalues) == 10
