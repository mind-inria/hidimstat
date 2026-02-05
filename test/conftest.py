import numpy as np
import pytest
from packaging.version import parse
from sklearn import __version__ as sklearn_version
from sklearn.utils.estimator_checks import (
    check_estimator as sklearn_check_estimator,
)

from hidimstat._utils.scenario import multivariate_simulation

SKLEARN_LT_1_6 = parse(sklearn_version).minor < 6

try:
    import matplotlib
except ImportError:
    matplotlib = None


def pytest_configure(config):  # noqa: ARG001
    """Use Agg so that no figures pop up."""
    if matplotlib is not None:
        matplotlib.use("Agg", force=True)


@pytest.fixture
def data_generator(
    n_samples,
    n_features,
    support_size,
    rho,
    seed,
    value,
    signal_noise_ratio,
    rho_serial,
):
    """
    Generate simulated data for testing.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    n_features : int
        Number of features in the dataset.
    support_size : int
        Number of important features (features with non-zero coefficients).
    rho : float
        Correlation coefficient between features.
    seed : int
        Random seed for reproducibility.
    value : float
        Value to be used for non-zero coefficients.
    signal_noise_ratio : float
        Signal-to-noise ratio.
    rho_serial : float
        Time correlation coefficient in the noise component.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target vector.
    important_features : ndarray
        Indices of features with non-zero coefficients.
    not_important_features : ndarray
        Indices of features with zero coefficients.
    """
    X, y, beta, _noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        rho=rho,
        value=value,
        signal_noise_ratio=signal_noise_ratio,
        rho_serial=rho_serial,
        shuffle=False,
        seed=seed,
    )
    important_features = np.where(beta != 0)[0]
    not_important_features = np.where(beta == 0)[0]
    return X, y, important_features, not_important_features


def check_estimator(
    estimators, return_expected_failed_checks, valid: bool = True
):
    """Yield a valid or invalid sklearn estimators check.

    ONLY USED FOR sklearn<1.6

    As some of estimators do not comply
    with sklearn recommendations
    (cannot fit Numpy arrays, do input validation in the constructor...)
    we cannot directly use
    sklearn.utils.estimator_checks.check_estimator.

    So this is a home made generator that yields an estimator instance
    along with a
    - valid check from sklearn: those should stay valid
    - or an invalid check that is known to fail.

    See this section rolling-your-own-estimator in
    the scikit-learn doc for more info:
    https://scikit-learn.org/stable/developers/develop.html

    Parameters
    ----------
    estimators : list of estimator object
        Estimator instance to check.

    valid : bool, default=True
        Whether to return only the valid checks or not.
    """
    # TODO (sklearn >= 1.6.0) remove this function
    if not SKLEARN_LT_1_6:  # pragma: no cover
        raise RuntimeError(
            "Use dedicated sklearn utilities to test estimators."
        )

    for est in estimators:
        expected_failed_checks = return_expected_failed_checks(est)

        for e, check in sklearn_check_estimator(
            estimator=est, generate_only=True
        ):
            if not valid and check.func.__name__ in expected_failed_checks:
                yield e, check, check.func.__name__
            if valid and check.func.__name__ not in expected_failed_checks:
                yield e, check, check.func.__name__
