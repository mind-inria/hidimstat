import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from hidimstat import CFI, BasePerturbation


def test_cfi(linear_scenario):
    """Test the Conditional Feature Importance algorithm on a linear scenario."""
    X, y, beta = linear_scenario
    important_features = np.where(beta != 0)[0]
    non_important_features = np.where(beta == 0)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    imputation_model = LinearRegression()

    cfi = CFI(
        estimator=regression_model,
        imputation_model_continuous=clone(imputation_model),
        imputation_model_categorical=LogisticRegression(),
        n_permutations=20,
        method="predict",
        random_state=0,
        n_jobs=1,
    )

    cfi.fit(
        X_train,
        groups=None,
        var_type="auto",
    )
    vim = cfi.importance(X_test, y_test)

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
    cfi = CFI(
        estimator=regression_model,
        imputation_model_continuous=clone(imputation_model),
        n_permutations=20,
        method="predict",
        random_state=0,
        n_jobs=1,
    )
    cfi.fit(
        X_train_df,
        groups=groups,
        var_type="continuous",
    )
    # warnings because we don't consider the name of columns of pandas
    with pytest.warns(UserWarning, match="X does not have valid feature names, but"):
        vim = cfi.importance(X_test_df, y_test)

    importance = vim["importance"]
    assert importance[0].mean() > importance[1].mean()

    # Classification scenario
    y_clf = np.where(y > 0, 1, 0)

    _, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, random_state=0)
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train_clf)

    cfi = CFI(
        estimator=logistic_model,
        imputation_model_continuous=clone(imputation_model),
        n_permutations=20,
        random_state=0,
        n_jobs=1,
        method="predict_proba",
        loss=log_loss,
    )
    cfi.fit(
        X_train,
        groups=None,
        var_type=["continuous"] * X.shape[1],
    )
    vim = cfi.importance(X_test, y_test_clf)


def test_raises_value_error(
    linear_scenario,
):
    """Test for the ValueError raised by the Conditional Feature Importance
    algorithm."""
    X, y, _ = linear_scenario

    # Predict method not recognized
    with pytest.raises(ValueError):
        fitted_model = LinearRegression().fit(X, y)
        predict_method = "unknown method"
        CFI(
            estimator=fitted_model,
            method=predict_method,
        )

    # Not fitted estimator
    with pytest.raises(NotFittedError):
        cfi = CFI(
            estimator=LinearRegression(),
            method="predict",
        )

    # Not fitted imputation model with predict and importance methods
    with pytest.raises(ValueError, match="The estimator is not fitted."):
        fitted_model = LinearRegression().fit(X, y)
        cfi = CFI(
            estimator=fitted_model,
            method="predict",
        )
        cfi.predict(X)
    with pytest.raises(ValueError, match="The estimator is not fitted."):
        fitted_model = LinearRegression().fit(X, y)
        cfi = CFI(
            estimator=fitted_model,
            method="predict",
        )
        cfi.importance(X, y)

    with pytest.raises(
        ValueError, match="The estimators require to be fit before to use them"
    ):
        fitted_model = LinearRegression().fit(X, y)
        cfi = CFI(
            estimator=fitted_model,
            method="predict",
        )
        BasePerturbation.fit(cfi, X, y)
        cfi.importance(X, y)
