import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.utils.estimator_checks import parametrize_with_checks

from hidimstat import D0CRT, LOCO, LOCOCV, PFI, PFICV
from hidimstat._utils.scenario import multivariate_simulation

from .conftest import SKLEARN_LT_1_6, check_estimator, fitted_linear_regression


def fitted_ridged_cv():
    """Return a fitted RidgeCV model."""
    X, y, _, _ = multivariate_simulation(
        n_samples=500,
        n_features=50,
    )
    return RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X, y)


def list_fitted_ridge_cv():
    """Return a list of fitted RidgeCV models for each fold."""
    X, y, _, _ = multivariate_simulation(
        n_samples=500,
        n_features=50,
    )
    model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    cv = KFold(n_splits=2, shuffle=True, random_state=0)
    return [
        clone(model).fit(X[train_index], y[train_index])
        for train_index, _ in cv.split(X)
    ]


def expected_failed_checks(estimator):
    if isinstance(estimator, PFI):
        return {}
    elif isinstance(estimator, PFICV):
        failed_checks = {}
        if isinstance(estimator.estimators, list):
            failed_checks = {}
        return failed_checks


ESTIMATORS_TO_CHECK = [
    PFI(
        estimator=fitted_linear_regression(),
    ),
    PFI(
        estimator=LinearRegression(),
    ),
    PFICV(
        estimators=RidgeCV(),
        cv=KFold(n_splits=2),
    ),
    PFICV(
        estimators=fitted_ridged_cv(),
        cv=KFold(n_splits=2),
    ),
    PFICV(
        estimators=list_fitted_ridge_cv(),
        cv=KFold(n_splits=2, shuffle=True, random_state=0),
    ),
    LOCO(
        estimator=LinearRegression(),
    ),
    LOCO(
        estimator=fitted_linear_regression(),
    ),
    LOCOCV(
        estimators=RidgeCV(),
        cv=KFold(n_splits=2),
    ),
    LOCOCV(
        estimators=fitted_ridged_cv(),
        cv=KFold(n_splits=2),
    ),
    D0CRT(estimator=LassoCV(n_jobs=1), screening_threshold=None),
]

if SKLEARN_LT_1_6:

    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(
            estimators=ESTIMATORS_TO_CHECK,
            return_expected_failed_checks=expected_failed_checks,
        ),
    )
    def test_check_estimator_sklearn_valid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)

    @pytest.mark.xfail(reason="invalid checks should fail")
    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(
            estimators=ESTIMATORS_TO_CHECK,
            valid=False,
            return_expected_failed_checks=expected_failed_checks,
        ),
    )
    def test_check_estimator_sklearn_invalid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)

else:

    @parametrize_with_checks(
        estimators=ESTIMATORS_TO_CHECK,
        expected_failed_checks=expected_failed_checks,
    )
    def test_check_estimator_sklearn(estimator, check):
        check(estimator)
