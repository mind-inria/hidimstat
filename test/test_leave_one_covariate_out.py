from functools import partial

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_1samp
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, train_test_split

from hidimstat import LOCO, LOCOCV, loco_importance
from hidimstat._utils.scenario import multivariate_simulation
from hidimstat.base_perturbation import BasePerturbation
from hidimstat.statistical_tools.multiple_testing import fdp_power


def test_loco():
    """Test the Leave-One-Covariate-Out algorithm on a linear scenario."""
    X, y, beta, noise = multivariate_simulation(
        n_samples=150,
        n_features=200,
        support_size=10,
        shuffle=False,
        seed=42,
    )
    important_features = np.where(beta != 0)[0]
    non_important_features = np.where(beta == 0)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    loco = LOCO(
        estimator=regression_model,
        method="predict",
        features_groups=None,
        n_jobs=1,
    )

    loco.fit(
        X_train,
        y_train,
    )
    importance = loco.importance(X_test, y_test)

    assert importance.shape == (X.shape[1],)
    assert (
        importance[important_features].mean()
        > importance[non_important_features].mean()
    )

    # Same with groups and a pd.DataFrame
    groups = {
        "group_0": [f"col_{i}" for i in important_features],
        "the_group_1": [f"col_{i}" for i in non_important_features],
    }
    X_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, random_state=0
    )
    regression_model.fit(X_train_df, y_train)
    loco = LOCO(
        estimator=regression_model,
        method="predict",
        features_groups=groups,
        n_jobs=1,
    )
    loco.fit(
        X_train_df,
        y_train,
    )
    # warnings because we doesn't consider the name of columns of pandas
    with pytest.warns(
        UserWarning, match="X does not have valid feature names, but"
    ):
        importance = loco.importance(X_test_df, y_test)

    assert importance[0].mean() > importance[1].mean()

    # Classification case
    y_clf = np.where(y > np.median(y), 1, 0)
    _, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, random_state=0)
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train_clf)

    loco_clf = LOCO(
        estimator=logistic_model,
        method="predict_proba",
        features_groups={
            "group_0": important_features,
            "the_group_1": non_important_features,
        },
        loss=log_loss,
        n_jobs=1,
    )
    loco_clf.fit(
        X_train,
        y_train_clf,
    )
    importance_clf = loco_clf.importance(X_test, y_test_clf)

    assert importance_clf.shape == (2,)
    assert importance_clf[0].mean() > importance_clf[1].mean()


def test_raises_value_error():
    """Test for error when model does not have predict_proba or predict."""
    X, y, beta, noise = multivariate_simulation(
        n_samples=150,
        n_features=200,
        support_size=10,
        shuffle=False,
        seed=42,
    )
    # Not fitted estimator
    with pytest.raises(NotFittedError):
        loco = LOCO(
            estimator=LinearRegression(),
            method="predict",
        )

    # Not fitted sub-model when calling importance and predict
    with pytest.raises(ValueError, match="The class is not fitted."):
        fitted_model = LinearRegression().fit(X, y)
        loco = LOCO(
            estimator=fitted_model,
            method="predict",
        )
        loco.importance(X, None)
    with pytest.raises(ValueError, match="The class is not fitted."):
        fitted_model = LinearRegression().fit(X, y)
        loco = LOCO(
            estimator=fitted_model,
            method="predict",
        )
        loco.importance(X, y)

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
        match="The statistical test doesn't provide the correct dimension.",
    ):
        fitted_model = LinearRegression().fit(X, y)
        loco = LOCO(
            estimator=fitted_model,
            statistical_test=partial(ttest_1samp, popmean=0, axis=0),
        ).fit(X, y)
        loco.importance(X, y)


def test_loco_function():
    """Test the function of LOCO algorithm on a linear scenario."""
    X, y, beta, noise = multivariate_simulation(
        n_samples=150,
        n_features=100,
        support_size=10,
        shuffle=False,
        seed=42,
    )
    important_features = np.where(beta != 0)[0]
    non_important_features = np.where(beta == 0)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    selection, importance, pvalue = loco_importance(
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
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(500, 50, 5, 0.1, 0, 2.0, 8, 0.0)],
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
    X, y, important_features, not_important_features = data_generator

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
