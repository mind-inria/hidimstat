"""
Test the desparsified_lasso module
"""

import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from scipy.linalg import toeplitz
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV

from hidimstat.desparsified_lasso import DesparsifiedLasso, desparsified_lasso
from hidimstat._utils.scenario import multivariate_simulation


def test_desparsified_lasso():
    """Testing the procedure on a simulation with no structure and
    a support of size 1. Computing 99% confidence bounds and checking
    that they contains the true parameter vector."""

    n_samples, n_features = 52, 50
    support_size = 1
    signal_noise_ratio = 50
    rho = 0.0

    X, y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        signal_noise_ratio=signal_noise_ratio,
        rho=rho,
        shuffle=False,
        seed=10,
    )
    expected_pval_corr = np.ones_like(beta) * 0.5
    expected_pval_corr[beta != 0] = 0.0

    desparsified_lasso = DesparsifiedLasso(confidence=0.99, random_state=2).fit(X, y)
    importances = desparsified_lasso.importance(X, y)

    assert_almost_equal(importances, beta, decimal=1)
    assert_equal(desparsified_lasso.confidence_bound_min_ < beta, True)
    assert_equal(desparsified_lasso.confidence_bound_max_ > beta, True)
    assert_almost_equal(desparsified_lasso.pvalues_corr_, expected_pval_corr, decimal=1)

    desparsified_lasso = DesparsifiedLasso(dof_ajdustement=True, confidence=0.99).fit(
        X, y
    )
    importances = desparsified_lasso.importance(X, y)

    assert_almost_equal(importances, beta, decimal=1)
    assert_equal(desparsified_lasso.confidence_bound_min_ < beta, True)
    assert_equal(desparsified_lasso.confidence_bound_max_ > beta, True)
    assert_almost_equal(desparsified_lasso.pvalues_corr_, expected_pval_corr, decimal=1)


def test_desparsified_group_lasso():
    """Testing the procedure on a simulation with no structure and
    a support of size 2. Computing one-sided p-values, we want
    low p-values for the features of the support and p-values
    close to 0.5 for the others."""

    n_samples = 50
    n_features = 100
    n_target = 10
    support_size = 2
    signal_noise_ratio = 5000
    rho_serial = 0.9
    corr = toeplitz(np.geomspace(1, rho_serial ** (n_target - 1), n_target))
    multitasklassoCV = MultiTaskLassoCV(
        eps=1e-2,
        fit_intercept=False,
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        tol=1e-4,
        max_iter=5000,
        random_state=1,
        n_jobs=1,
    )

    X, y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_target,
        support_size=support_size,
        rho_serial=rho_serial,
        signal_noise_ratio=signal_noise_ratio,
        seed=10,
    )

    desparsified_lasso = DesparsifiedLasso(
        lasso_cv=multitasklassoCV, covariance=corr
    ).fit(X, y)
    importances = desparsified_lasso.importance(X, y)

    assert_almost_equal(importances, beta, decimal=1)

    expected_pval_corr = np.ones_like(beta[:, 0]) * 0.5
    expected_pval_corr[beta[:, 0] != 0] = 0.0

    assert_almost_equal(importances, beta, decimal=1)
    assert_almost_equal(desparsified_lasso.pvalues_corr_, expected_pval_corr, decimal=1)

    desparsified_lasso = DesparsifiedLasso(lasso_cv=multitasklassoCV, test="F").fit(
        X, y
    )
    importances = desparsified_lasso.importance(X, y)

    assert_almost_equal(importances, beta, decimal=1)
    assert_almost_equal(desparsified_lasso.pvalues_corr_, expected_pval_corr, decimal=1)

    # Testing error is raised when the covariance matrix has wrong shape
    bad_cov = np.delete(corr, 0, axis=1)
    # np.testing.assert_raises(
    # ValueError, desparsified_lasso, X=X, y=y, multioutput=True, covariance=bad_cov
    # )
    desparsified_lasso = DesparsifiedLasso(
        lasso_cv=multitasklassoCV, covariance=bad_cov
    ).fit(X, y)
    with pytest.raises(ValueError):
        desparsified_lasso.importance(X, y)

    with pytest.raises(AssertionError, match="Unknown test 'r2'"):
        DesparsifiedLasso(lasso_cv=multitasklassoCV, covariance=bad_cov, test="r2").fit(
            X, y
        )


def test_exception():
    n_samples = 50
    n_features = 100
    n_target = 10
    support_size = 2
    signal_noise_ratio = 5000
    rho_serial = 0.9
    corr = toeplitz(np.geomspace(1, rho_serial ** (n_target - 1), n_target))
    multitasklassoCV = MultiTaskLassoCV(
        eps=1e-2,
        fit_intercept=False,
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        tol=1e-4,
        max_iter=5000,
        random_state=1,
        n_jobs=1,
    )

    X, y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_target,
        support_size=support_size,
        rho_serial=rho_serial,
        signal_noise_ratio=signal_noise_ratio,
        seed=10,
    )

    with pytest.raises(
        AssertionError, match="lasso needs to be a Lasso or a MultiTaskLasso"
    ):
        DesparsifiedLasso(lasso=RandomForestClassifier())
    with pytest.raises(
        AssertionError, match="lasso_cv needs to be a LassoCV or a MultiTaskLassoCV"
    ):
        DesparsifiedLasso(lasso_cv=RandomForestClassifier())
    with pytest.raises(AssertionError, match="Unknown test 'r2'"):
        DesparsifiedLasso(test="r2")
    desparsified_lasso = DesparsifiedLasso(lasso_cv=multitasklassoCV)
    with pytest.raises(
        ValueError,
        match="The Desparsified Lasso requires to be fit before any analysis",
    ):
        desparsified_lasso.importance(X, y)
    desparsified_lasso.sigma_hat_ = []
    with pytest.raises(
        ValueError,
        match="The Desparsified Lasso requires to be fit before any analysis",
    ):
        desparsified_lasso.importance(X, y)

    desparsified_lasso = DesparsifiedLasso(lasso_cv=multitasklassoCV).fit(X, y)
    with pytest.raises(ValueError, match="Unknown test 'r2'"):
        desparsified_lasso.test = "r2"
        desparsified_lasso.importance(X, y)


def test_warning():
    n_samples, n_features = 52, 50
    support_size = 1
    signal_noise_ratio = 50
    rho = 0.0

    X, y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        signal_noise_ratio=signal_noise_ratio,
        rho=rho,
        shuffle=False,
        seed=10,
    )
    desparsified_lasso = DesparsifiedLasso(
        lasso_cv=LassoCV(cv=KFold(n_splits=2), max_iter=10)
    )
    with pytest.warns(Warning, match="'max_iter' has been increased to"):
        with pytest.warns(Warning, match="cv won't be used"):
            desparsified_lasso.fit_importance(X, y, cv=[])


def test_function_not_center():
    "Test function and not centered"
    n_samples, n_features = 52, 50
    support_size = 1
    signal_noise_ratio = 50
    rho = 0.0

    X, y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        signal_noise_ratio=signal_noise_ratio,
        rho=rho,
        shuffle=False,
        seed=10,
    )
    selection, importances, pvalues = desparsified_lasso(X, y, centered=False)
