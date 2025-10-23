"""
Test the dcrt module
"""

from hidimstat.dcrt import dcrt_zero
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
            verbose=True,
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
            verbose=True,
            statistic="test",
            random_state=2024,
        )

    # Checking with and without screening
    results_no_screening = dcrt_zero(
        X, y, screening=False, verbose=True, statistic="residual", random_state=2024
    )
    results_screening = dcrt_zero(
        X, y, screening=True, verbose=True, statistic="residual", random_state=2024
    )
    assert len(results_no_screening[1]) == 10
    assert len(results_no_screening[2]) == 10
    assert len(results_screening[1]) == 10
    assert len(results_screening[2]) == 10

    # Checking with random estimated coefficients for the features
    rng = np.random.RandomState(2024)
    estimated_coefs = rng.rand(10)

    res = dcrt_zero(
        X,
        y,
        estimated_coef=estimated_coefs,
        screening=False,
        verbose=True,
        statistic="residual",
        random_state=2024,
    )
    assert len(res[1]) == 10
    assert len(res[2]) == 10

    # Checking with refit
    results_refit = dcrt_zero(
        X,
        y,
        refit=True,
        screening=False,
        verbose=True,
        statistic="residual",
        random_state=2024,
    )
    assert len(results_refit[1]) == 10
    assert len(results_refit[2]) == 10

    # Checking with scaled statistics
    results_scaled = dcrt_zero(
        X,
        y,
        scaled_statistics=True,
        screening=False,
        verbose=True,
        statistic="residual",
        random_state=2024,
    )
    assert len(results_scaled[1]) == 10
    assert len(results_scaled[2]) == 10

    # Checking without verbose (returns only selected indices)
    results_no_verbose = dcrt_zero(
        X,
        y,
        scaled_statistics=True,
        screening=False,
        verbose=False,
        statistic="residual",
        random_state=2024,
    )
    assert len(results_no_verbose) <= 10

    # Checking with use_cv
    results_use_cv = dcrt_zero(
        X,
        y,
        use_cv=True,
        screening=False,
        verbose=True,
        statistic="residual",
        random_state=2024,
    )
    assert len(results_use_cv[1]) == 10
    assert len(results_use_cv[2]) == 10

    # Checking with a provided covariance matrix
    cov = LedoitWolf().fit(X)

    results_covariance = dcrt_zero(
        X,
        y,
        Sigma_X=cov.covariance_,
        screening=False,
        verbose=True,
        statistic="residual",
        random_state=2024,
    )
    assert len(results_covariance[1]) == 10
    assert len(results_covariance[2]) == 10


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
            verbose=True,
            loss="test",
            random_state=2024,
            statistic="randomforest",
        )

    res = dcrt_zero(
        X,
        y,
        screening=False,
        verbose=True,
        statistic="randomforest",
        problem_type="regression",
        random_state=2024,
    )
    assert len(res[1]) == 10
    assert len(res[2]) == 10


def test_dcrt_RF_classification():
    """
    This function tests the dcrt function using the Random Forest learner
    """
    X, y = make_classification(n_samples=100, n_features=10, random_state=2024)
    res = dcrt_zero(
        X,
        y,
        screening=False,
        verbose=True,
        statistic="randomforest",
        problem_type="classification",
        random_state=2024,
    )
    assert len(res[1]) == 10
    assert len(res[2]) == 10
