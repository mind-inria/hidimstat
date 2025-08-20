import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from hidimstat import LOCO, BasePerturbation


def test_loco(linear_scenario):
    """Test the Leave-One-Covariate-Out algorithm on a linear scenario."""
    X, y, beta = linear_scenario
    important_features = np.where(beta != 0)[0]
    non_important_features = np.where(beta == 0)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    loco = LOCO(
        estimator=regression_model,
        method="predict",
        n_jobs=1,
    )

    loco.fit(
        X_train,
        y_train,
        groups=None,
    )
    vim = loco.importance(X_test, y_test)

    importance = vim["importance"]
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
    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, random_state=0)
    regression_model.fit(X_train_df, y_train)
    loco = LOCO(
        estimator=regression_model,
        method="predict",
        n_jobs=1,
    )
    loco.fit(
        X_train_df,
        y_train,
        groups=groups,
    )
    # warnings because we doesn't considere the name of columns of pandas
    with pytest.warns(UserWarning, match="X does not have valid feature names, but"):
        vim = loco.importance(X_test_df, y_test)

    importance = vim["importance"]
    assert importance[0].mean() > importance[1].mean()

    # Classification case
    y_clf = np.where(y > np.median(y), 1, 0)
    _, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, random_state=0)
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train_clf)

    loco_clf = LOCO(
        estimator=logistic_model,
        method="predict_proba",
        n_jobs=1,
        loss=log_loss,
    )
    loco_clf.fit(
        X_train,
        y_train_clf,
        groups={"group_0": important_features, "the_group_1": non_important_features},
    )
    vim_clf = loco_clf.importance(X_test, y_test_clf)

    importance_clf = vim_clf["importance"]
    assert importance_clf.shape == (2,)
    assert importance[0].mean() > importance[1].mean()


def test_raises_value_error(
    linear_scenario,
):
    """Test for error when model does not have predict_proba or predict."""
    X, y, _ = linear_scenario
    # Not fitted estimator
    with pytest.raises(NotFittedError):
        loco = LOCO(
            estimator=LinearRegression(),
            method="predict",
        )

    # Not fitted sub-model when calling importance and predict
    with pytest.raises(ValueError, match="The estimator is not fitted."):
        fitted_model = LinearRegression().fit(X, y)
        loco = LOCO(
            estimator=fitted_model,
            method="predict",
        )
        loco.predict(X)
    with pytest.raises(ValueError, match="The estimator is not fitted."):
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
