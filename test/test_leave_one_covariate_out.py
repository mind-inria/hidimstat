from functools import partial

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_1samp
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from hidimstat import LOCO, LOCOCV, loco_importance
from hidimstat._utils.scenario import multivariate_simulation
from hidimstat.base_perturbation import BasePerturbation
from hidimstat.statistical_tools.multiple_testing import fdp_power


def run_loco(
    X,
    y,
    estimator,
    features_groups=None,
    method="predict",
    loss=mean_squared_error,
):
    """Test the Leave-One-Covariate-In algorithm on a linear scenario."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    estimator.fit(X_train, y_train)

    loco = LOCO(
        estimator=estimator,
        method=method,
        loss=loss,
        features_groups=features_groups,
        n_jobs=1,
    )

    loco.fit(
        X_train,
        y_train,
    )
    loco.importance(X_test, y_test)

    return loco


@pytest.mark.parametrize(
    "n_samples, n_features, n_targets, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(100, 20, None, 4, 0, 42, 1, 10, 0)],
    ids=["basic data"],
)
def test_loco(data_generator):
    """Test the Leave-One-Covariate-Out algorithm on a linear scenario."""
    X, y, beta, _ = data_generator
    important_features = np.zeros(X.shape[1], dtype=bool)
    important_features[beta] = True
    non_important_features = ~important_features

    loco = run_loco(X=X, y=y, estimator=LinearRegression())
    importance = loco.importances_

    assert importance.shape == (X.shape[1],)
    assert (
        importance[important_features].mean()
        > importance[non_important_features].mean()
    )

    # Same with groups and a pd.DataFrame
    groups = {
        "group_0": [f"col_{i}" for i in beta],
        "the_group_1": [
            f"col_{i}" for i in np.arange(X.shape[1])[non_important_features]
        ],
    }
    X_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
    loco = run_loco(
        X=X_df, y=y, estimator=LinearRegression(), features_groups=groups
    )
    importance = loco.importances_

    assert importance[0].mean() > importance[1].mean()

    # Classification case
    y_clf = np.where(y > np.median(y), 1, 0)
    loco_clf = run_loco(
        X=X,
        y=y_clf,
        estimator=LogisticRegression(),
        features_groups={
            "group_0": beta,
            "the_group_1": np.arange(X.shape[1])[non_important_features],
        },
        method="predict_proba",
        loss=log_loss,
    )
    importance_clf = loco_clf.importances_

    assert importance_clf.shape == (2,)
    assert importance_clf[0].mean() > importance_clf[1].mean()


@pytest.mark.parametrize(
    "n_samples, n_features, n_targets, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(100, 20, None, 4, 0, 42, 1, 10, 0)],
    ids=["basic data"],
)
def test_raises_value_error(data_generator):
    """Test for error when model does not have predict_proba or predict."""
    X, y, _, _ = data_generator

    # Not fitted sub-model when calling importance and predict
    with pytest.raises(
        ValueError, match="This LOCO instance is not fitted yet"
    ):
        fitted_model = LinearRegression().fit(X, y)
        loco = LOCO(
            estimator=fitted_model,
            method="predict",
        )
        loco.importance(X, None)

    with pytest.raises(
        ValueError, match="The estimators require to be fit before to use them"
    ):
        fitted_model = LinearRegression().fit(X, y)
        loco = LOCO(
            estimator=fitted_model,
            method="predict",
        )
        BasePerturbation.fit(loco, X, y)
        loco.importance(X, y)

    with pytest.raises(
        AssertionError,
        match="The statistical test doesn't provide the correct dimension",
    ):
        fitted_model = LinearRegression().fit(X, y)
        loco = LOCO(
            estimator=fitted_model,
            statistical_test=partial(ttest_1samp, popmean=0, axis=0),
        ).fit(X, y)
        loco.importance(X, y)


@pytest.mark.parametrize(
    "n_samples, n_features, n_targets, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(100, 20, None, 4, 0, 42, 1, 10, 0)],
    ids=["basic data"],
)
def test_loco_function(data_generator):
    """Test the function of LOCO algorithm on a linear scenario."""
    X, y, beta, _ = data_generator
    important_features = np.zeros(X.shape[1], dtype=bool)
    important_features[beta] = True
    non_important_features = ~important_features

    X_train, _, y_train, _ = train_test_split(X, y, random_state=0)

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    _, importance, _ = loco_importance(
        regression_model,
        X,
        y,
        method="predict",
        n_jobs=1,
    )

    assert importance.shape == (X.shape[1],)
    assert (
        importance[important_features].mean()
        > importance[non_important_features].mean()
    )


@pytest.mark.parametrize(
    "n_samples, n_features, n_targets, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(500, 50, None, 5, 0.1, 0, 2.0, 8, 0.0)],
    ids=["default data"],
)
def test_loco_cv(data_generator):
    """
    Test that LOCO with cross-validated estimator works as expected. In particular,
        - Empirical FDP is below the target FDR level
        - Power is above 0.8, which is an arbitrary threshold

    Note: even though the only the expected FDP should be controlled, in practice
    the simulation setting is simple enough to satisfy this stronger condition.
    """
    X, y, important_features, _ = data_generator

    model = RidgeCV()
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    loco_cv = LOCOCV(
        estimators=model,
        cv=cv,
        n_jobs=5,
    )
    loco_cv.fit(X, y)
    loco_cv.importance(X, y)

    alpha = 0.1
    selected = loco_cv.fdr_selection(fdr=alpha)
    gt_mask = np.zeros(X.shape[1], dtype=int)
    gt_mask[important_features] = 1
    fdp, power = fdp_power(selected=selected, ground_truth=gt_mask)
    assert fdp < alpha
    assert power > 0.8
