from functools import partial

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_1samp
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

from hidimstat import LOCI, LOCICV, loci_importance
from hidimstat._utils.scenario import multivariate_simulation
from hidimstat.base_perturbation import BasePerturbation
from hidimstat.statistical_tools.multiple_testing import fdp_power


def run_loci(
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

    loci = LOCI(
        estimator=estimator,
        method=method,
        loss=loss,
        features_groups=features_groups,
        n_jobs=1,
    )

    loci.fit(
        X_train,
        y_train,
    )
    loci.importance(X_test, y_test)

    return loci


@pytest.mark.parametrize(
    "n_samples, n_features, n_targets, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(100, 20, None, 4, 0, 42, 1, 10, 0)],
    ids=["basic data"],
)
def test_loci(data_generator):
    """Test the Leave-One-Covariate-In algorithm on a linear scenario."""
    X, y, beta, _ = data_generator
    important_features = np.zeros(X.shape[1], dtype=bool)
    important_features[beta] = True
    non_important_features = ~important_features

    loci = run_loci(X=X, y=y, estimator=LinearRegression())
    importance = loci.importances_

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
    loci = run_loci(
        X=X_df, y=y, estimator=LinearRegression(), features_groups=groups
    )
    importance = loci.importances_

    assert importance[0].mean() > importance[1].mean()

    # Classification case
    y_clf = np.where(y > np.median(y), 1, 0)
    loci_clf = run_loci(
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
    importance_clf = loci_clf.importances_

    assert importance_clf.shape == (2,)
    assert importance_clf[0].mean() > importance_clf[1].mean()


def test_multiclass_loci():
    """Test LOCI in a multiclass classification setup."""
    important_features = 4
    n_features = 20

    X, y = make_classification(
        n_samples=100,
        n_features=n_features,
        n_informative=important_features,
        n_classes=3,
        random_state=42,
        shuffle=False,
    )
    groups = {
        "important": list(range(important_features)),
        "unimportant": list(range(important_features, n_features)),
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    logistic_model = LogisticRegression(random_state=0)
    logistic_model.fit(X_train, y_train)

    loci_clf = LOCI(
        estimator=logistic_model,
        method="predict_proba",
        features_groups=groups,
        loss=log_loss,
        n_jobs=1,
    )
    loci_clf.fit(
        X_train,
        y_train,
    )
    importance_clf = loci_clf.importance(X_test, y_test)

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
        ValueError, match="This LOCI instance is not fitted yet"
    ):
        fitted_model = LinearRegression().fit(X, y)
        loci = LOCI(
            estimator=fitted_model,
            method="predict",
        )
        loci.importance(X, None)

    with pytest.raises(
        ValueError, match="The estimators require to be fit before to use them"
    ):
        fitted_model = LinearRegression().fit(X, y)
        loci = LOCI(
            estimator=fitted_model,
            method="predict",
        )
        BasePerturbation.fit(loci, X, y)
        loci.importance(X, y)

    with pytest.raises(
        AssertionError,
        match="The statistical test doesn't provide the correct dimension",
    ):
        fitted_model = LinearRegression().fit(X, y)
        loci = LOCI(
            estimator=fitted_model,
            statistical_test=partial(ttest_1samp, popmean=0, axis=0),
        ).fit(X, y)
        loci.importance(X, y)


@pytest.mark.parametrize(
    "n_samples, n_features, n_targets, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(100, 20, None, 4, 0, 42, 1, 10, 0)],
    ids=["basic data"],
)
def test_loci_function(data_generator):
    """Test the function of LOCI algorithm on a linear scenario."""
    X, y, beta, _ = data_generator
    important_features = np.zeros(X.shape[1], dtype=bool)
    important_features[beta] = True
    non_important_features = ~important_features

    X_train, _, y_train, _ = train_test_split(X, y, random_state=0)

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    _, importance, _ = loci_importance(
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
    [(500, 50, None, 5, 0.0, 0, 2.0, 8, 0.0)],
    ids=["default data"],
)
def test_loci_cv(data_generator):
    """
    Test that LOCI with cross-validated estimator works as expected. In particular,
        - Empirical FDP is below the target FDR level
        - Power is above 0.8, which is an arbitrary threshold

    Note: even though the only the expected FDP should be controlled, in practice
    the simulation setting is simple enough to satisfy this stronger condition.
    """
    X, y, important_features, _ = data_generator

    model = RidgeCV()
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    loci_cv = LOCICV(
        estimators=model,
        cv=cv,
        n_jobs=5,
    )
    loci_cv.fit(X, y)
    loci_cv.importance(X, y)

    alpha = 0.2
    selected = loci_cv.fdr_selection(fdr=alpha)
    gt_mask = np.zeros(X.shape[1], dtype=int)
    gt_mask[important_features] = 1
    fdp, power = fdp_power(selected=selected, ground_truth=gt_mask)
    assert fdp < alpha
    assert power >= 0.8
