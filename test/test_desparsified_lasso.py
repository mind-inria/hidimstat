"""
Test the desparsified_lasso module
"""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.linalg import toeplitz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import KFold

from hidimstat._utils.scenario import multivariate_simulation
from hidimstat.desparsified_lasso import DesparsifiedLasso, desparsified_lasso


def test_desparsified_lasso():
    """
    Test desparsified lasso on a simple simulation with no structure and
    a support of size 5.
     - Test that the confidence intervals contain the true beta 70% of the time. This
    threshold is arbitrary.
     - Test that the empirical false discovery proportion is below the target FDR
    Although this is not guaranteed (control is only in expectation), the scenario
    is simple enough for the test to pass
    - Test that the true discovery proportion is above 80%, this threshold is arbitrary
    """

    n_samples, n_features = 400, 40
    support_size = 5
    signal_noise_ratio = 32
    rho = 0.0
    confidence = 0.9
    alpha = 1 - confidence

    X, y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        signal_noise_ratio=signal_noise_ratio,
        rho=rho,
        shuffle=False,
    )

    desparsified_lasso = DesparsifiedLasso(confidence=confidence).fit(X, y)
    _ = desparsified_lasso.importance()
    # Check that beta is within the confidence intervals
    correct_interval = np.sum(
        (beta >= desparsified_lasso.confidence_bound_min_)
        & (beta <= desparsified_lasso.confidence_bound_max_)
    )
    assert correct_interval >= int(0.7 * n_features)

    # Check p-values for important and non-important features
    important = beta != 0
    non_important = beta == 0
    tp = np.sum(desparsified_lasso.pvalues_corr_[important] < alpha)
    fp = np.sum(desparsified_lasso.pvalues_corr_[non_important] < alpha)
    assert fp / np.sum(non_important) <= alpha
    assert tp / np.sum(important) >= 0.8

    desparsified_lasso = DesparsifiedLasso(
        dof_ajdustement=True, confidence=confidence
    ).fit(X, y)
    _ = desparsified_lasso.importance()

    # Check that beta is within the confidence intervals
    correct_interval = np.sum(
        (beta >= desparsified_lasso.confidence_bound_min_)
        & (beta <= desparsified_lasso.confidence_bound_max_)
    )
    assert correct_interval >= int(0.7 * n_features)

    # Check p-values for important and non-important features
    tp = np.sum(desparsified_lasso.pvalues_corr_[important] < alpha)
    fp = np.sum(desparsified_lasso.pvalues_corr_[non_important] < alpha)
    assert fp / np.sum(non_important) <= alpha
    assert tp / np.sum(important) >= 0.8


def test_desparsified_group_lasso():
    """
    Testing the procedure on a simulation with no structure and a support of size 2.
     - Test that the empirical false discovery proportion is below the target FDR
    Although this is not guaranteed (control is only in expectation), the scenario
    is simple enough for the test to pass.
     - Test that the true discovery proportion is above 80%, this threshold is arbitrary
    """

    n_samples = 400
    n_features = 40
    n_target = 10
    support_size = 5
    signal_noise_ratio = 32
    rho_serial = 0.9
    alpha = 0.1

    corr = toeplitz(np.geomspace(1, rho_serial ** (n_target - 1), n_target))
    multitasklassoCV = MultiTaskLassoCV(
        eps=1e-2,
        fit_intercept=False,
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        tol=1e-4,
        max_iter=50,
        random_state=1,
        n_jobs=1,
    )

    X, y, beta, _ = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_target,
        support_size=support_size,
        rho_serial=rho_serial,
        signal_noise_ratio=signal_noise_ratio,
    )

    with pytest.warns(Warning, match="'max_iter' has been increased to "):
        desparsified_lasso = DesparsifiedLasso(
            model_y=multitasklassoCV, covariance=corr, save_model_x=True
        ).fit(X, y)
    importances = desparsified_lasso.importance()

    assert_almost_equal(importances, beta, decimal=1)

    important = beta[:, 0] != 0
    non_important = beta[:, 0] == 0

    assert_almost_equal(importances, beta, decimal=1)
    tp = np.sum(desparsified_lasso.pvalues_corr_[important] < alpha)
    fp = np.sum(desparsified_lasso.pvalues_corr_[non_important] < alpha)
    assert fp / np.sum(non_important) <= alpha
    assert tp / np.sum(important) >= 0.8
    assert (
        desparsified_lasso.clf_ is not None
        and len(desparsified_lasso.clf_) == n_features
    )

    desparsified_lasso = DesparsifiedLasso(model_y=multitasklassoCV, test="F").fit(X, y)
    importances = desparsified_lasso.importance()

    assert_almost_equal(importances, beta, decimal=1)
    tp = np.sum(desparsified_lasso.pvalues_corr_[important] < alpha)
    fp = np.sum(desparsified_lasso.pvalues_corr_[non_important] < alpha)
    assert fp / np.sum(non_important) <= alpha
    assert tp / np.sum(important) >= 0.8
    assert desparsified_lasso

    # Testing error is raised when the covariance matrix has wrong shape
    bad_cov = np.delete(corr, 0, axis=1)
    # np.testing.assert_raises(
    # ValueError, desparsified_lasso, X=X, y=y, multioutput=True, covariance=bad_cov
    # )
    desparsified_lasso = DesparsifiedLasso(
        model_y=multitasklassoCV, covariance=bad_cov
    ).fit(X, y)
    with pytest.raises(ValueError):
        desparsified_lasso.importance()

    with pytest.raises(AssertionError, match="Unknown test 'r2'"):
        DesparsifiedLasso(model_y=multitasklassoCV, covariance=bad_cov, test="r2").fit(
            X, y
        )


def test_exception():
    n_samples = 50
    n_features = 100
    n_target = 10
    support_size = 2
    signal_noise_ratio = 50
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
        DesparsifiedLasso(model_x=RandomForestClassifier())
    with pytest.raises(
        AssertionError, match="lasso_cv needs to be a LassoCV or a MultiTaskLassoCV"
    ):
        DesparsifiedLasso(model_y=RandomForestClassifier())
    with pytest.raises(AssertionError, match="Unknown test 'r2'"):
        DesparsifiedLasso(test="r2")
    desparsified_lasso = DesparsifiedLasso(model_y=multitasklassoCV)
    with pytest.raises(
        ValueError,
        match="The Desparsified Lasso requires to be fit before any analysis",
    ):
        desparsified_lasso.importance()

    desparsified_lasso = DesparsifiedLasso(model_y=multitasklassoCV).fit(X, y)
    with pytest.raises(ValueError, match="Unknown test 'r2'"):
        desparsified_lasso.test = "r2"
        desparsified_lasso.importance()

    desparsified_lasso = DesparsifiedLasso(model_y=multitasklassoCV).fit(X, y)
    with pytest.warns(Warning, match="X won't be used."):
        desparsified_lasso.importance(X=X)
    with pytest.warns(Warning, match="y won't be used."):
        desparsified_lasso.importance(y=y)


def test_function_not_center():
    """Test function when the data don't need to be centered"""
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
