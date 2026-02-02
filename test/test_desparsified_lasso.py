"""
Test the desparsified_lasso module
"""

from copy import copy

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.linalg import toeplitz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
from sklearn.model_selection import KFold

from hidimstat._utils.scenario import multivariate_simulation
from hidimstat.desparsified_lasso import (
    DesparsifiedLasso,
    desparsified_lasso_importance,
    reid,
)
from hidimstat.statistical_tools.multiple_testing import fdp_power


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
    n_samples, n_features = 500, 50
    support_size = 5
    signal_noise_ratio = 32
    rho = 0.0
    confidence = 0.9
    alpha = 1 - confidence
    # Tolerance for the FDP and power test
    test_tol = 0.05

    fdp_list = []
    powr_list = []
    fdp_dof_list = []
    powr_dof_list = []
    for seed in range(10):
        X, y, beta, noise = multivariate_simulation(
            n_samples=n_samples,
            n_features=n_features,
            support_size=support_size,
            signal_noise_ratio=signal_noise_ratio,
            rho=rho,
            shuffle=False,
            seed=seed,
        )

        desparsified_lasso = DesparsifiedLasso(
            confidence=confidence, random_state=seed
        ).fit(X, y)
        _ = desparsified_lasso.importance()
        # Check that beta is within the confidence intervals
        correct_interval = np.sum(
            (beta >= desparsified_lasso.confidence_bound_min_)
            & (beta <= desparsified_lasso.confidence_bound_max_)
        )
        assert correct_interval >= int(0.7 * n_features)

        # Check p-values for important and non-important features
        important = beta != 0
        # Run two sided test for coverage
        selected = desparsified_lasso.fdr_selection(
            fdr=alpha, two_tailed_test=False
        )
        fdp, power = fdp_power(selected=selected, ground_truth=important)
        fdp_list.append(fdp)
        powr_list.append(power)

        desparsified_lasso = DesparsifiedLasso(
            dof_ajdustement=True, confidence=confidence, random_state=seed
        ).fit(X, y)
        _ = desparsified_lasso.importance()

        # Check that beta is within the confidence intervals
        correct_interval = np.sum(
            (beta >= desparsified_lasso.confidence_bound_min_)
            & (beta <= desparsified_lasso.confidence_bound_max_)
        )

        # Check p-values for important and non-important features
        selected = desparsified_lasso.fdr_selection(
            fdr=alpha, two_tailed_test=True
        )
        fdp, power = fdp_power(selected=selected, ground_truth=important)
        assert correct_interval >= int(0.7 * n_features)
        fdp_dof_list.append(fdp)
        powr_dof_list.append(power)

    assert np.mean(fdp_list) <= alpha + test_tol
    assert np.mean(powr_list) >= 0.8 - test_tol
    assert np.mean(fdp_dof_list) <= alpha + test_tol
    assert np.mean(powr_dof_list) >= 0.8 - test_tol


def test_desparsified_group_lasso():
    """
    Testing the procedure on a simulation with no structure and a support of size 2.
     - Test that the empirical FWER is below the target FWER
     - Test that the true discovery proportion is above 80%, this threshold is arbitrary
    """
    n_samples = 500
    n_features = 50
    n_target = 10
    support_size = 5
    signal_noise_ratio = 32
    rho_serial = 0.9
    alpha = 0.1
    # Small tolerance for the test
    test_tol = 0.05

    fd_list = []
    power_list = []
    fd_ftest_list = []
    power_ftest_list = []
    for seed in range(10):
        corr = toeplitz(
            np.geomspace(1, rho_serial ** (n_target - 1), n_target)
        )
        multi_task_lasso_cv = MultiTaskLassoCV(
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
            seed=seed,
        )

        with pytest.warns(Warning, match="'max_iter' has been increased to "):
            desparsified_lasso = DesparsifiedLasso(
                estimator=multi_task_lasso_cv,
                covariance=corr,
                save_model_x=True,
                random_state=seed,
            ).fit(X, y)
        importances = desparsified_lasso.importance()

        assert_almost_equal(importances, beta, decimal=1)

        important = beta[:, 0] != 0

        assert_almost_equal(importances, beta, decimal=1)
        selected = desparsified_lasso.fwer_selection(
            fwer=alpha, two_tailed_test=False
        )
        fdp, power = fdp_power(selected=selected, ground_truth=important)
        fd_list.append(fdp > 0)
        power_list.append(power)
        assert (
            desparsified_lasso.clf_ is not None
            and len(desparsified_lasso.clf_) == n_features
        )

        desparsified_lasso = DesparsifiedLasso(
            estimator=multi_task_lasso_cv, test="F", random_state=seed
        ).fit(X, y)
        importances = desparsified_lasso.importance()

        assert_almost_equal(importances, beta, decimal=1)
        selected = desparsified_lasso.fwer_selection(
            fwer=alpha, two_tailed_test=False
        )
        fdp, power = fdp_power(selected=selected, ground_truth=important)
        fd_ftest_list.append(fdp > 0)
        power_ftest_list.append(power)

        # Test with two_tailed_test=True for coverage
        selected_two_tailed = desparsified_lasso.fdr_selection(
            fdr=alpha, two_tailed_test=True
        )
        assert selected_two_tailed.shape == (n_features,)

        # Testing error is raised when the covariance matrix has wrong shape
        bad_cov = np.delete(corr, 0, axis=1)
        desparsified_lasso = DesparsifiedLasso(
            estimator=multi_task_lasso_cv, covariance=bad_cov
        ).fit(X, y)
        with pytest.raises(ValueError):
            desparsified_lasso.importance()

        with pytest.raises(AssertionError, match="Unknown test 'r2'"):
            DesparsifiedLasso(
                estimator=multi_task_lasso_cv, covariance=bad_cov, test="r2"
            ).fit(X, y)

    assert np.mean(fd_list) <= alpha + test_tol
    assert np.mean(power_list) >= 0.8 - test_tol
    assert np.mean(fd_ftest_list) <= alpha + test_tol
    assert np.mean(power_ftest_list) >= 0.8 - test_tol


def test_exception():
    """Test exception of Desparsified Lasso"""
    n_samples = 50
    n_features = 100
    n_target = 10
    support_size = 2
    signal_noise_ratio = 50
    rho_serial = 0.9
    corr = toeplitz(np.geomspace(1, rho_serial ** (n_target - 1), n_target))
    multi_task_lasso_cv = MultiTaskLassoCV(
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
        AssertionError,
        match="model_x needs to be a Lasso, LassoCV, or a MultiTaskLasso",
    ):
        DesparsifiedLasso(model_x=RandomForestClassifier())
    with pytest.raises(
        AssertionError,
        match="lasso_cv needs to be a LassoCV or a MultiTaskLassoCV",
    ):
        DesparsifiedLasso(estimator=RandomForestClassifier())
    with pytest.raises(AssertionError, match="Unknown test 'r2'"):
        DesparsifiedLasso(test="r2")
    desparsified_lasso = DesparsifiedLasso(estimator=multi_task_lasso_cv)
    with pytest.raises(
        ValueError,
        match="The Desparsified Lasso requires to be fit before any analysis",
    ):
        desparsified_lasso.importance()

    desparsified_lasso = DesparsifiedLasso(estimator=multi_task_lasso_cv).fit(
        X, y
    )
    with pytest.raises(ValueError, match="Unknown test 'r2'"):
        desparsified_lasso.test = "r2"
        desparsified_lasso.importance()

    desparsified_lasso = DesparsifiedLasso(estimator=multi_task_lasso_cv).fit(
        X, y
    )
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
    selection, importances, pvalues = desparsified_lasso_importance(
        X, y, centered=False
    )


def test_reid():
    """Estimating noise standard deviation in two scenarios.
    First scenario: no structure and a support of size 2.
    Second scenario: no structure and an empty support.
    """
    n_samples, n_features = 100, 20
    signal_noise_ratio = 2.0

    # First expe
    # ##########
    support_size = 2

    X, y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        rho=0.25,
        signal_noise_ratio=signal_noise_ratio,
        seed=0,
    )
    lasso_cv = LassoCV(n_jobs=1).fit(X, y)
    residual = lasso_cv.predict(X) - y

    # max_iter=1 to get a better coverage
    sigma_hat = reid(lasso_cv.coef_, residual, tolerance=1e-3)
    expected_sigma = support_size / signal_noise_ratio
    error_relative = np.abs(sigma_hat - expected_sigma) / expected_sigma
    assert error_relative < 0.3

    # Second expe
    # ###########
    support_size = 0

    X, y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        signal_noise_ratio=signal_noise_ratio,
        seed=2,
    )
    lasso_cv = LassoCV(n_jobs=1).fit(X, y)
    residual = lasso_cv.predict(X) - y

    sigma_hat = reid(lasso_cv.coef_, residual)
    expected_sigma = 1.0  # when there is no signal, the variance is 1.0
    error_relative = np.abs(sigma_hat - expected_sigma) / expected_sigma
    assert error_relative < 0.2


def test_group_reid():
    """Estimating (temporal) noise covariance matrix in two scenarios.
    First scenario: no data structure and a support of size 2.
    Second scenario: no data structure and an empty support.
    """
    n_samples = 100
    n_features = 20
    n_target = 50
    signal_noise_ratio = 3.0
    rho_serial = 0.9
    random_state = np.random.default_rng(1)

    # First expe
    # ##########
    support_size = 2
    X, y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_target,
        support_size=support_size,
        signal_noise_ratio=signal_noise_ratio,
        rho_serial=rho_serial,
        rho=0.0,
        seed=0,
    )
    corr = toeplitz(np.geomspace(1, rho_serial ** (n_target - 1), n_target))
    cov = support_size / signal_noise_ratio * corr

    lasso_cv = MultiTaskLassoCV(n_jobs=1).fit(X, y)
    residual = lasso_cv.predict(X) - y

    # max_iter=1 to get a better coverage
    cov_hat = reid(
        lasso_cv.coef_,
        residual,
        multioutput=True,
        tolerance=1e-3,
    )
    error_relative = np.abs(cov_hat - cov) / cov
    assert np.max(error_relative) < 0.3

    cov_hat = reid(
        lasso_cv.coef_,
        residual,
        multioutput=True,
        method="AR",
    )
    error_relative = np.abs(cov_hat - cov) / cov
    assert np.max(error_relative) < 0.3

    cov_hat = reid(
        lasso_cv.coef_,
        residual,
        multioutput=True,
        stationary=False,
    )
    error_relative = np.abs(cov_hat - cov) / cov
    assert np.max(error_relative) > 0.3


def test_group_reid_2():
    """Estimating (temporal) noise covariance matrix in two scenarios.
    First scenario: no data structure and a support of size 2.
    Second scenario: no data structure and an empty support.
    """
    n_samples = 100
    n_features = 20
    n_target = 50
    signal_noise_ratio = 1.0
    rho_serial = 0.9

    # Second expe
    # ###########
    support_size = 0
    X, y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_target,
        rho=0.25,
        support_size=support_size,
        signal_noise_ratio=signal_noise_ratio,
        rho_serial=rho_serial,
        seed=4,
    )
    corr = toeplitz(
        rho_serial ** np.arange(0, n_target)
    )  # covariance matrix of time
    cov = 1.0 * corr

    lasso_cv = MultiTaskLassoCV(n_jobs=1).fit(X, y)
    residual = lasso_cv.predict(X) - y

    cov_hat = reid(lasso_cv.coef_, residual, multioutput=True)
    error_relative = np.abs(cov_hat - cov) / cov
    assert np.max(error_relative) < 0.3

    cov_hat = reid(lasso_cv.coef_, residual, multioutput=True, method="AR")
    error_relative = np.abs(cov_hat - cov) / cov
    assert np.max(error_relative) < 0.3

    cov_hat = reid(
        lasso_cv.coef_, residual, multioutput=True, stationary=False
    )
    error_relative = np.abs(cov_hat - cov) / cov
    assert np.max(error_relative) > 0.3


def test_reid_exception():
    """Test for testing the exceptions on the arguments of reid function"""
    n_samples, n_features = 100, 20
    n_target = 50
    signal_noise_ratio = 1.0
    rho_serial = 0.9

    # First expe
    # ##########
    support_size = 2

    X, y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_target,
        support_size=support_size,
        signal_noise_ratio=signal_noise_ratio,
        rho_serial=rho_serial,
    )
    with pytest.raises(
        ValueError, match="Unknown method for estimating the covariance matrix"
    ):
        _, _ = reid(X, y, method="test", multioutput=True)
    with pytest.raises(
        ValueError,
        match="The AR method is not compatible with the non-stationary",
    ):
        _, _ = reid(X, y, method="AR", stationary=False, multioutput=True)
    with pytest.raises(
        ValueError, match="The requested AR order is to high with"
    ):
        _, _ = reid(X, y, method="AR", order=1e4, multioutput=True)


@pytest.fixture(scope="module")
def dl_y1d_test_data():
    """
    Fixture to generate test data for single-target regression problem. We use LassoCV
    as the estimator with shuffling in the cross-validation to introduce randomness in
    the training process.
    """
    X, y, beta, _ = multivariate_simulation(
        n_samples=100,
        n_features=20,
        # n_targets=1,
        support_size=2,
        signal_noise_ratio=8,
        rho_serial=0.5,
    )
    dl = DesparsifiedLasso(
        estimator=LassoCV(cv=KFold(n_splits=3, shuffle=True)),
        model_x=LassoCV(cv=KFold(n_splits=3, shuffle=True)),
        preconfigure_model_x_path=False,
    )
    return X, y, dl


def test_dl_repeatibility(dl_y1d_test_data):
    """
    Test that multiple calls of .importance() when DL is seeded provide deterministic
    results.
    """
    X, y, dl = dl_y1d_test_data
    dl.random_state = 0
    y_ = y.ravel()
    dl.fit(X, y_)
    importance = dl.importance(X, y_)
    # repeat
    importance_repeat = dl.importance(X, y_)
    assert np.array_equal(importance, importance_repeat)


def test_dl_randomness_with_none(dl_y1d_test_data):
    """
    Test that multiple calls of .importance() when DL has random_state=None

    Note:
        Only the fit is affected by random_state (through the random_state of the
        different estimators). There is no randomness in the importance computation.
    """
    X, y, dl = dl_y1d_test_data
    dl_1 = copy(dl)
    dl_1.random_state = None
    dl_1.fit(X, y)
    importance = dl_1.importance(X, y)

    # refit
    dl_1.fit(X, y)
    importance_refit = dl_1.importance(X, y)
    assert not np.array_equal(importance, importance_refit)

    # Reproducibility
    dl_repro = copy(dl)
    dl_repro.fit(X, y)
    importance_repro = dl_repro.importance(X, y)
    assert not np.array_equal(importance, importance_repro)


def test_dl_reproducibility_with_integer(dl_y1d_test_data):
    """
    Test that multiple calls of .importance() when DL has random_state=0
    """
    X, y, dl = dl_y1d_test_data
    dl_1 = copy(dl)
    dl_1.random_state = 0
    importance = dl_1.fit_importance(X, y)

    # Refit
    importance_refit = dl_1.fit_importance(X, y)
    assert np.array_equal(importance, importance_refit)

    # Reproducibility
    dl_repro = copy(dl)
    dl_repro.random_state = 0
    dl_repro.fit(X, y)
    importance_repro = dl_repro.importance(X, y)
    assert np.array_equal(importance, importance_repro)


def test_dl_reproducibility_with_rng(dl_y1d_test_data):
    """
    Test that:
    1. Refit the same instance when using a rng as random_state provides different results.
    2. Refit different instances with the same rng provides same result.

    Note:
        Only the fit is affected by random_state (through the random_state of the
        different estimators). There is no randomness in the importance computation.
    """
    X, y, dl = dl_y1d_test_data
    dl_1 = copy(dl)
    dl_1.random_state = np.random.default_rng(0)
    importance_1 = dl_1.fit_importance(X, y)

    # refit
    dl_1.fit(X, y)
    importance_refit = dl_1.importance(X, y)
    assert not np.array_equal(importance_1, importance_refit)

    # refit repeatability
    dl_refit = copy(dl)
    dl_refit.random_state = np.random.default_rng(0)
    importance_refit = dl_refit.fit_importance(X, y)
    assert np.array_equal(importance_1, importance_refit)
