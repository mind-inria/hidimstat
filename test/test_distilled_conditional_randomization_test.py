"""
Test the dcrt module
"""

import numpy as np
import pytest
from numpy.random import RandomState
from sklearn.covariance import LedoitWolf
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold

from hidimstat import D0CRT, d0crt
from hidimstat._utils.regression import _alpha_max
from hidimstat._utils.scenario import multivariate_simulation


@pytest.fixture
def generate_regression_dataset(n=100, p=10, noise=0.2, seed=2024):
    X, y = make_regression(n_samples=n, n_features=p, noise=noise, random_state=seed)
    return X, y


def test_dcrt_lasso_screening(generate_regression_dataset):
    """
    Test for screening parameter and pvalue function
    """
    X, y = generate_regression_dataset
    # Checking with and without screening
    d0crt_no_screening = D0CRT(
        estimator=LassoCV(n_jobs=1),
        screening_threshold=None,
    )
    pvalue_no_screening = d0crt_no_screening.fit_importance(X, y)
    sv_no_screening = d0crt_no_screening.selection(threshold_pvalue=0.05)
    d0crt_screening = D0CRT(
        estimator=LassoCV(n_jobs=1),
        screening_threshold=10,
    )
    pvalue_screening = d0crt_screening.fit_importance(X, y)
    sv_screening = d0crt_screening.selection(threshold_pvalue=0.05)
    assert np.sum(d0crt_no_screening.importances_ != 0) <= 10
    assert np.sum(d0crt_screening.importances_ != 0) <= 10
    assert len(sv_no_screening) <= 10
    assert len(pvalue_no_screening) == 10
    assert len(d0crt_no_screening.importances_) == 10
    assert len(sv_screening) <= 10
    assert len(pvalue_screening) == 10
    assert len(d0crt_screening.importances_) == 10

    # Checking with scaled statistics
    d0crt_no_screening = D0CRT(
        estimator=LassoCV(n_jobs=1),
        screening_threshold=None,
        scaled_statistics=True,
    )
    d0crt_no_screening.fit_importance(X, y)
    pvalue_no_screening = d0crt_no_screening.importance(X, y)
    sv_no_screening = d0crt_no_screening.selection(threshold_pvalue=0.05)
    assert len(sv_no_screening) <= 10
    assert len(pvalue_no_screening) == 10
    assert len(d0crt_no_screening.importances_) == 10


def test_dcrt_lasso_with_estimed_coefficient(generate_regression_dataset):
    """
    Test the estimated coefficient parameter
    """
    X, y = generate_regression_dataset
    # Checking with random estimated coefficients for the features
    rng = np.random.RandomState(2025)
    estimated_coefs = rng.rand(10)

    d0crt = D0CRT(
        estimator=LassoCV(n_jobs=1),
        estimated_coef=estimated_coefs,
        screening_threshold=None,
    )
    d0crt.fit(X, y)
    pvalue = d0crt.importance(X, y)
    sv = d0crt.selection(threshold_pvalue=0.05)
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.importances_) == 10


def test_dcrt_lasso_with_refit(generate_regression_dataset):
    """
    Test the refit parameter
    """
    X, y = generate_regression_dataset
    # Checking with refit
    d0crt_refit = D0CRT(
        estimator=LassoCV(n_jobs=1),
        refit=True,
        screening_threshold=None,
    )
    pvalue = d0crt_refit.fit_importance(X, y)
    sv = d0crt_refit.selection(threshold_pvalue=0.05)
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt_refit.importances_) == 10


def test_dcrt_lasso_with_no_cv(generate_regression_dataset):
    """
    Test the parameters to the Lasso of x-distillation
    """
    X, y = generate_regression_dataset
    # Checking with use_cv
    d0crt_use_cv = D0CRT(
        estimator=LassoCV(n_jobs=1),
        lasso_screening=LassoCV(n_jobs=1, fit_intercept=False),
        screening_threshold=None,
    )
    pvalue = d0crt_use_cv.fit_importance(X, y)
    sv = d0crt_use_cv.selection(threshold_pvalue=0.05)
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt_use_cv.importances_) == 10


def test_dcrt_lasso_with_covariance(generate_regression_dataset):
    """
    Test dcrt with provide covariance matrix
    """
    X, y = generate_regression_dataset
    # Checking with a provided covariance matrix
    cov = LedoitWolf().fit(X)

    d0crt_covariance = D0CRT(
        estimator=LassoCV(n_jobs=1),
        sigma_X=cov.covariance_,
        screening_threshold=None,
    )
    pvalue = d0crt_covariance.fit_importance(X, y)
    sv = d0crt_covariance.selection(threshold_pvalue=0.05)
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt_covariance.importances_) == 10


def test_dcrt_lasso_center():
    """
    Test for not center the data
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.2, random_state=2024)
    d0crt = D0CRT(
        estimator=LassoCV(n_jobs=1),
        centered=False,
        screening_threshold=None,
        scaled_statistics=True,
    )
    pvalue = d0crt.fit_importance(X, y)
    sv = d0crt.selection(threshold_pvalue=0.05)
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.importances_) == 10


def test_dcrt_lasso_refit():
    """
    This function tests the dcrt function using the Lasso learner and refit
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.2, random_state=2024)
    d0crt = D0CRT(
        estimator=LassoCV(n_jobs=1),
        refit=True,
        fit_y=True,
        scaled_statistics=True,
    )
    pvalue = d0crt.fit_importance(X, y)
    sv = d0crt.selection(threshold_pvalue=0.05)
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.importances_) == 10


def test_dcrt_lasso_no_selection():
    """
    This function tests the dcrt function using the Lasso learner
    with distillation y when precomputed coefficients are provided.
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.8, random_state=20)
    d0crt = D0CRT(estimator=LassoCV(n_jobs=1), estimated_coef=np.ones(10) * 10)
    d0crt.fit(X, y)
    assert np.all(d0crt.selection_set_)


def test_dcrt_distillation_x_different():
    """
    This function tests the dcrt function using the Lasso learner
    with distillation x using different argument
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.8, random_state=20)
    d0crt = D0CRT(
        estimator=Lasso(alpha=0.5 * _alpha_max(X, y), fit_intercept=False),
        scaled_statistics=True,
    )
    pvalue = d0crt.fit_importance(X, y)
    sv = d0crt.selection(threshold_pvalue=0.05)
    assert np.where(d0crt.importances_ != 0)[0].shape[0] <= 10
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.importances_) == 10


def test_dcrt_distillation_y_different():
    """
    This function tests the dcrt function using the Lasso learner
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.8, random_state=20)
    d0crt = D0CRT(
        estimator=LassoCV(n_jobs=1),
        model_distillation_x=Lasso(),
        scaled_statistics=True,
    )
    pvalue = d0crt.fit_importance(X, y)
    sv = d0crt.selection(threshold_pvalue=0.05)
    assert np.where(d0crt.importances_ != 0)[0].shape[0] <= 10
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.importances_) == 10


def test_dcrt_lasso_fit_with_no_cv():
    """
    Test the dcrt function using the Lasso learner
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.2, random_state=2024)
    d0crt = D0CRT(
        estimator=LassoCV(n_jobs=1),
        fit_y=True,
        lasso_screening=LassoCV(n_jobs=1),
        screening_threshold=None,
        scaled_statistics=True,
    )
    pvalue = d0crt.fit_importance(X, y)
    sv = d0crt.selection(threshold_pvalue=0.05)
    assert np.sum(d0crt.importances_ != 0) <= 10
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.importances_) == 10


def test_dcrt_RF_regression():
    """
    This function tests the dcrt function using the Random Forest learner
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.2, random_state=2024)

    d0crt = D0CRT(
        estimator=RandomForestRegressor(n_estimators=100, random_state=2026, n_jobs=1),
        method="predict",
        screening_threshold=None,
        scaled_statistics=True,
    )
    pvalue = d0crt.fit_importance(X, y)
    sv = d0crt.selection(threshold_pvalue=0.05)
    assert np.where(d0crt.importances_ != 0)[0].shape[0] <= 10
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.importances_) == 10


def test_dcrt_RF_classification():
    """
    This function tests the dcrt function using the Random Forest learner
    """
    X, y = make_classification(n_samples=100, n_features=10, random_state=2024)
    d0crt = D0CRT(
        estimator=RandomForestClassifier(n_estimators=100, random_state=2026, n_jobs=1),
        method="predict_proba",
        screening_threshold=None,
        scaled_statistics=True,
    )
    pvalue = d0crt.fit_importance(X, y)
    sv = d0crt.selection(threshold_pvalue=0.05)
    assert np.where(d0crt.importances_ != 0)[0].shape[0] <= 10
    assert len(sv) <= 10
    assert len(pvalue) == 10
    assert len(d0crt.importances_) == 10


def test_exception_not_fitted():
    """Test if an exception is raised when the method is not fitted"""
    X, y = make_classification(n_samples=100, n_features=10, random_state=2024)
    d0crt = D0CRT(
        estimator=RandomForestClassifier(n_estimators=100, random_state=2026, n_jobs=1),
        method="predict_proba",
        screening_threshold=None,
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
        screening_threshold=None,
    )
    d0crt.fit(X, y)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    with pytest.warns(UserWarning, match="cv won't be used"):
        _ = d0crt.fit_importance(X, y, cv=cv)


def test_dcrt_invalid_lasso_screening(generate_regression_dataset):
    """
    Test that passing a non-Lasso model to lasso_screening raises a ValueError.
    """
    X, y = generate_regression_dataset

    d0crt = D0CRT(
        estimator=LassoCV(n_jobs=1),
        lasso_screening=RandomForestRegressor(n_estimators=10, random_state=0),
        screening_threshold=10,
    )
    with pytest.raises(
        ValueError, match="lasso_model must be an instance of Lasso or LassoCV"
    ):
        d0crt.fit(X, y)


def test_function_d0crt():
    """Test the d0crt function"""
    X, y = make_regression(
        n_samples=100,
        n_features=10,
        noise=0.2,
    )
    sv, importances, pvalues = d0crt(LassoCV(n_jobs=1), X, y)
    assert len(sv) <= 10
    assert len(importances) == 10
    assert len(pvalues) == 10


@pytest.fixture(scope="module")
def d0crt_test_data():
    X, y, _, _ = multivariate_simulation(
        n_samples=150,
        n_features=20,
        support_size=10,
        rho=0,
        value=1,
        signal_noise_ratio=2,
        rho_serial=0,
        shuffle=False,
        seed=0,
    )
    dcrt_default_parameters = {
        "estimator": LassoCV(cv=KFold(shuffle=True)),
        "screening_threshold": None,
        "model_distillation_x": LassoCV(n_alphas=10, cv=KFold(shuffle=True)),
    }
    return X, y, dcrt_default_parameters


def test_d0crt_repeatability(d0crt_test_data):
    """
    Test that different instances of D0CRT with the same random state provide
    deterministic results.
    """
    X, y, dcrt_default_parameters = d0crt_test_data
    d0crt_1 = D0CRT(**dcrt_default_parameters, random_state=0)
    d0crt_1.fit(X, y)
    vim_1 = d0crt_1.importance(X, y)
    vim_repeat = d0crt_1.importance(X, y)
    assert np.array_equal(vim_1, vim_repeat)


def test_d0crt_randomness_with_none(d0crt_test_data):
    """
    Test that different random states provide different results and that
    random_state=None produces randomness.
    """
    X, y, dcrt_default_parameters = d0crt_test_data
    # Fixed random state
    d0crt_fixed = D0CRT(**dcrt_default_parameters, random_state=0)
    d0crt_fixed.fit(X, y)
    vim_fixed = d0crt_fixed.importance(X, y)

    # Different random state
    d0crt_none_state = D0CRT(**dcrt_default_parameters, random_state=None)
    d0crt_none_state.fit(X, y)
    vim_none_state_1 = d0crt_none_state.importance(X, y)
    assert not np.array_equal(vim_fixed, vim_none_state_1)

    d0crt_none_state.fit(X, y)
    vim_none_state_2 = d0crt_none_state.importance(X, y)
    assert not np.array_equal(vim_none_state_1, vim_none_state_2)


def test_d0crt_reproducibility_with_integer(d0crt_test_data):
    """
    Test that different instances of D0CRT with the same random state provide
    deterministic results.
    """
    X, y, dcrt_default_parameters = d0crt_test_data
    d0crt_1 = D0CRT(**dcrt_default_parameters, random_state=0)
    d0crt_1.fit(X, y)
    vim_1 = d0crt_1.importance(X, y)

    d0crt_2 = D0CRT(**dcrt_default_parameters, random_state=0)
    d0crt_2.fit(X, y)
    vim_2 = d0crt_2.importance(X, y)
    assert np.array_equal(vim_1, vim_2)


def test_d0crt_reproducibility_with_rng(d0crt_test_data):
    """
    Test that:
     1. Mmultiple calls of .importance() when CFI has random_state=rng are random
     2. refit with same rng provides same result
    """
    X, y, dcrt_default_parameters = d0crt_test_data
    rng = np.random.default_rng(0)
    d0crt_1 = D0CRT(**dcrt_default_parameters, random_state=rng)
    d0crt_1.fit(X, y)
    vim_1 = d0crt_1.importance(X, y)

    d0crt_1.fit(X, y)
    vim_2 = d0crt_1.importance(X, y)
    assert not np.array_equal(vim_1, vim_2)

    rng = np.random.default_rng(0)
    d0crt_2 = D0CRT(**dcrt_default_parameters, random_state=rng)
    d0crt_2.fit(X, y)
    vim_3 = d0crt_2.importance(X, y)
    assert np.array_equal(vim_1, vim_3)


def test_d0crt_linear():
    """
    This function tests the D0CRT on a linear simulation, to see if it selects the
    correct features.
    It checks if the importances measured are larger for the features with non-zero
    coefficients and that the selected features correspond to the true informative
    features.
    """
    X, y, coef = make_regression(
        n_samples=100,
        n_features=10,
        noise=0.0,
        random_state=0,
        coef=True,
        n_informative=5,
        shuffle=False,
    )
    important_ids = np.where(coef != 0)[0]
    d0crt = D0CRT(
        estimator=LassoCV(n_jobs=1, n_alphas=10),
        screening_threshold=90,
    )
    importances = d0crt.fit_importance(X, y)
    sv = d0crt.selection(threshold_pvalue=0.05)

    assert np.mean(importances[important_ids]) > np.mean(importances[~important_ids])
    assert np.array_equal(np.where(sv)[0], important_ids)


def test_d0crt_rf():
    """
    This function tests the D0CRT on a random forest model.
    """
    X, y, coef = make_regression(
        n_samples=200,
        n_features=10,
        noise=0.0,
        random_state=0,
        coef=True,
        n_informative=5,
        shuffle=False,
    )
    important_ids = np.where(coef != 0)[0]
    d0crt = D0CRT(
        estimator=RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=1),
        screening_threshold=None,
        random_state=0,
    )
    importances = d0crt.fit_importance(X, y)
    sv = d0crt.selection(threshold_pvalue=0.05)

    assert np.mean(importances[important_ids]) > np.mean(importances[~important_ids])
    assert np.array_equal(np.where(sv)[0], important_ids)
