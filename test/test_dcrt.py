"""
Test the dcrt module
"""

from hidimstat.dcrt import dcrt_zero, dcrt_pvalue
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.datasets import make_regression, make_classification
import pytest


def test_dcrt_lasso():
    """
    This function tests the dcrt function using the Lasso learner
    """
    X, y = make_regression(n_samples=100, n_features=10, noise=0.2, random_state=2024)
    # Checking if a loss != 'least_square'
    with pytest.raises(Exception):
        _ = dcrt_zero(
            X,
            y,
            screening=False,
            statistic="residual",
            loss="test",
            random_state=2024,
        )

    # Checking for a different statistic
    with pytest.raises(Exception):
        _ = dcrt_zero(
            X,
            y,
            screening=False,
            statistic="test",
            random_state=2024,
        )

    # Checking with and without screening
    t_no_screening = dcrt_zero(
        X, y, screening=False, statistic="residual", random_state=2024
    )
    results_no_screening = dcrt_pvalue(t_no_screening, selection_only=False)
    t_screening = dcrt_zero(
        X, y, screening=True, statistic="residual", random_state=2024
    )
    results_screening = dcrt_pvalue(t_screening, selection_only=False)
    assert len(t_no_screening) == 10
    assert len(results_no_screening[0]) <= 10
    assert len(results_no_screening[1]) == 10
    assert len(t_screening) == 10
    assert len(results_screening[0]) <= 10
    assert len(results_screening[1]) == 10

    # Checking with random estimated coefficients for the features
    rng = np.random.RandomState(2024)
    estimated_coefs = rng.rand(10)

    t = dcrt_zero(
        X,
        y,
        estimated_coef=estimated_coefs,
        screening=False,
        statistic="residual",
        random_state=2024,
    )
    res = dcrt_pvalue(t, selection_only=False)
    assert len(t) == 10
    assert len(res[0]) <= 10
    assert len(res[1]) == 10

    # Checking with refit
    t_refit = dcrt_zero(
        X,
        y,
        refit=True,
        screening=False,
        statistic="residual",
        random_state=2024,
    )
    results_refit = dcrt_pvalue(t_refit, selection_only=False)
    assert len(t_refit) == 10
    assert len(results_refit[0]) <= 10
    assert len(results_refit[1]) == 10

    # Checking with scaled statistics
    t_scaled = dcrt_zero(
        X,
        y,
        scaled_statistics=True,
        screening=False,
        statistic="residual",
        random_state=2024,
    )
    results_scaled = dcrt_pvalue(t_scaled, selection_only=False)
    assert len(t_scaled) == 10
    assert len(results_scaled[0]) <= 10
    assert len(results_scaled[1]) == 10

    # Checking without verbose (returns only selected indices)
    t_no_verbose = dcrt_zero(
        X,
        y,
        scaled_statistics=True,
        screening=False,
        statistic="residual",
        random_state=2024,
    )
    results_no_verbose = dcrt_pvalue(t_no_verbose, selection_only=True)
    assert len(results_no_verbose) <= 10

    # Checking with use_cv
    t_use_cv = dcrt_zero(
        X,
        y,
        use_cv=True,
        screening=False,
        statistic="residual",
        random_state=2024,
    )
    results_use_cv = dcrt_pvalue(t_use_cv, selection_only=False)
    assert len(t_use_cv) == 10
    assert len(results_use_cv[0]) <= 10
    assert len(results_use_cv[1]) == 10

    # Checking with a provided covariance matrix
    cov = LedoitWolf().fit(X)

    t_covariance = dcrt_zero(
        X,
        y,
        sigma_X=cov.covariance_,
        screening=False,
        statistic="residual",
        random_state=2024,
    )
    results_covariance = dcrt_pvalue(t_covariance, selection_only=False)
    assert len(t_covariance) == 10
    assert len(results_covariance[0]) <= 10
    assert len(results_covariance[1]) == 10


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

    t = dcrt_zero(
        X,
        y,
        screening=False,
        statistic="randomforest",
        problem_type="regression",
        random_state=2024,
    )
    res = dcrt_pvalue(t, selection_only=False)
    assert len(t) == 10
    assert len(res[0]) <= 10
    assert len(res[1]) == 10


def test_dcrt_RF_classification():
    """
    This function tests the dcrt function using the Random Forest learner
    """
    X, y = make_classification(n_samples=100, n_features=10, random_state=2024)
    t = dcrt_zero(
        X,
        y,
        screening=False,
        statistic="randomforest",
        problem_type="classification",
        random_state=2024,
    )
    res = dcrt_pvalue(t, selection_only=False)
    assert len(t) == 10
    assert len(res[0]) <= 10
    assert len(res[1]) == 10
