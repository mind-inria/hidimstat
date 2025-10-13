"""
Test the noise_std module
"""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.linalg import toeplitz
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
from sklearn.model_selection import KFold

from hidimstat._utils.scenario import multivariate_simulation
from hidimstat.noise_std import empirical_snr, reid


def test_reid():
    """Estimating noise standard deviation in two scenarios.
    First scenario: no structure and a support of size 2.
    Second scenario: no structure and an empty support."""

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
    Second scenario: no data structure and an empty support."""

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
    Second scenario: no data structure and an empty support."""

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
    corr = toeplitz(rho_serial ** np.arange(0, n_target))  # covariance matrix of time
    cov = 1.0 * corr

    lasso_cv = MultiTaskLassoCV(n_jobs=1).fit(X, y)
    residual = lasso_cv.predict(X) - y

    cov_hat = reid(lasso_cv.coef_, residual, multioutput=True)
    error_relative = np.abs(cov_hat - cov) / cov
    assert np.max(error_relative) < 0.3

    cov_hat = reid(lasso_cv.coef_, residual, multioutput=True, method="AR")
    error_relative = np.abs(cov_hat - cov) / cov
    assert np.max(error_relative) < 0.3

    cov_hat = reid(lasso_cv.coef_, residual, multioutput=True, stationary=False)
    error_relative = np.abs(cov_hat - cov) / cov
    assert np.max(error_relative) > 0.3


def test_reid_exception():
    "Test for testing the exceptions on the arguments of reid function"
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
        ValueError, match="The AR method is not compatible with the non-stationary"
    ):
        _, _ = reid(X, y, method="AR", stationary=False, multioutput=True)
    with pytest.raises(ValueError, match="The requested AR order is to high with"):
        _, _ = reid(X, y, method="AR", order=1e4, multioutput=True)


def test_empirical_snr():
    """Computing empirical signal to noise ratio in presence of high level of
    noise from the target `y`, the data `X` and the true parameter vector `beta`
    in a simple scenario with a 1D data structure."""

    n_samples, n_features = 100, 20
    support_size = 10
    signal_noise_ratio_expected = 0.5

    X, y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        signal_noise_ratio=signal_noise_ratio_expected,
        seed=0,
    )

    signal_noise_ratio = empirical_snr(X, y, beta)

    assert_almost_equal(signal_noise_ratio, signal_noise_ratio_expected, decimal=2)


def test_empirical_snr_2():
    """Computing empirical signal to noise ratio from the target `y`,
    the data `X` and the true parameter vector `beta` in a simple
    scenario with a 1D data structure."""

    n_samples, n_features = 100, 20
    support_size = 10
    signal_noise_ratio_expected = 10.0

    X, y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        signal_noise_ratio=signal_noise_ratio_expected,
        seed=0,
    )

    signal_noise_ratio = empirical_snr(X, y, beta)

    assert_almost_equal(signal_noise_ratio, signal_noise_ratio_expected, decimal=0)
