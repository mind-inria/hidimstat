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
    results_no_screening = dcrt_zero(
        X, y, screening=False, statistic="residual", random_state=2024
    )
    vi_no_screening = dcrt_pvalue(
        results_no_screening[0],
        results_no_screening[1],
        results_no_screening[2],
        results_no_screening[3],
        selection_only=False,
    )
    results_screening = dcrt_zero(
        X, y, screening=True, statistic="residual", random_state=2024
    )
    vi_screening = dcrt_pvalue(
        results_screening[0],
        results_screening[1],
        results_screening[2],
        results_screening[3],
        selection_only=False,
    )
    assert np.sum(results_no_screening[0]) == 10
    assert np.sum(results_screening[0]) < 10
    assert len(vi_no_screening[0]) <= 10
    assert len(vi_no_screening[1]) == 10
    assert len(vi_no_screening[2]) == 10
    assert len(vi_screening[0]) <= 10
    assert len(vi_screening[1]) == 10
    assert len(vi_screening[2]) == 10

    # Checking with selection of variables (returns only selected indices)
    vi_no_verbose = dcrt_pvalue(
        results_screening[0],
        results_screening[1],
        results_screening[2],
        results_screening[3],
        selection_only=True,
    )
    assert len(vi_no_verbose) <= 10

    # Checking with scaled statistics
    vi_scaled = dcrt_pvalue(
        results_no_screening[0],
        results_no_screening[1],
        results_no_screening[2],
        results_no_screening[3],
        selection_only=False,
    )
    assert len(vi_scaled[0]) <= 10
    assert len(vi_scaled[1]) == 10
    assert len(vi_scaled[2]) == 10

    # Checking with random estimated coefficients for the features
    rng = np.random.RandomState(2024)
    estimated_coefs = rng.rand(10)

    results = dcrt_zero(
        X,
        y,
        estimated_coef=estimated_coefs,
        screening=False,
        statistic="residual",
        random_state=2024,
    )
    vi = dcrt_pvalue(
        results[0], results[1], results[2], results[3], selection_only=False
    )
    assert len(vi[0]) <= 10
    assert len(vi[1]) == 10
    assert len(vi[2]) == 10

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
        selection_only=False,
    )
    assert len(vi_refit[0]) <= 10
    assert len(vi_refit[1]) == 10
    assert len(vi_refit[2]) == 10

    # Checking with use_cv
    results_use_cv = dcrt_zero(
        X,
        y,
        use_cv=True,
        screening=False,
        statistic="residual",
        random_state=2024,
    )
    vi_use_cv = dcrt_pvalue(
        results_use_cv[0],
        results_use_cv[1],
        results_use_cv[2],
        results_use_cv[3],
        selection_only=False,
    )
    assert len(vi_use_cv[0]) <= 10
    assert len(vi_use_cv[1]) == 10
    assert len(vi_use_cv[1]) == 10

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
        selection_only=False,
    )
    assert len(vi_covariance[0]) <= 10
    assert len(vi_covariance[1]) == 10
    assert len(vi_covariance[1]) == 10


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
    vi = dcrt_pvalue(res[0], res[1], res[2], res[3], selection_only=False)
    assert len(vi[0]) <= 10
    assert len(vi[1]) == 10
    assert len(vi[2]) == 10


def test_dcrt_RF_classification():
    """
    This function tests the dcrt function using the Random Forest learner
    """
    with pytest.warns(UserWarning, match="Binary classification residuals"):
        X, y = make_classification(n_samples=100, n_features=10, random_state=2024)
        res = dcrt_zero(
            X,
            y,
            screening=False,
            statistic="randomforest",
            problem_type="classification",
            random_state=2024,
        )
    vi = dcrt_pvalue(res[0], res[1], res[2], res[3], selection_only=False)
    assert len(vi[0]) <= 10
    assert len(vi[1]) == 10
    assert len(vi[2]) == 10
