import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from hidimstat.permutation_importance import cpi


def test_cpi(linear_scenario):
    X, y, beta = linear_scenario
    important_features = np.where(beta != 0)[0]
    non_important_features = np.where(beta == 0)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    imputation_model = LinearRegression()

    importance, list_loss_j, loss_reference = cpi(
        X_train,
        X_test,
        y_test,
        estimator=regression_model,
        imputation_model=imputation_model,
        n_permutations=20,
        method="predict",
        random_state=0,
        n_jobs=1,
        groups=None,
    )

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
    imputation_model_list = {
        "group_0": clone(imputation_model),
        "the_group_1": clone(imputation_model),
    }
    regression_model.fit(X_train_df, y_train)
    importance, list_loss_j, loss_reference = cpi(
        X_train_df,
        X_test_df,
        y_test,
        estimator=regression_model,
        imputation_model=imputation_model_list,
        n_permutations=20,
        method="predict",
        random_state=0,
        n_jobs=1,
        groups=groups,
    )
    assert importance[0].mean() > importance[1].mean()

    # Classification scenario
    y_clf = np.where(y > 0, 1, 0)

    _, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, random_state=0)
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train_clf)

    importance, list_loss_j, loss_reference = cpi(
        X_train,
        X_test,
        y_test_clf,
        estimator=logistic_model,
        imputation_model=imputation_model,
        imputation_method="predict",
        n_permutations=20,
        random_state=0,
        n_jobs=1,
        method="predict_proba",
        loss=log_loss,
    )


def test_raises_value_error(
    linear_scenario,
):
    X, y, _ = linear_scenario

    # Predict method not recognized
    with pytest.raises(ValueError):
        fitted_model = LinearRegression().fit(X, y)
        predict_method = "unknown method"
        cpi(
            X,
            X,
            y,
            estimator=fitted_model,
            imputation_model=LinearRegression(),
            method=predict_method,
        )

    ## change dynamic
    # # Not fitted estimator
    # with pytest.raises(NotFittedError):
    #     cpi(
    #         X,
    #         X,
    #         y,
    #         estimator=LinearRegression(),
    #         imputation_model=LinearRegression(),
    #         method="predict",
    #     )

    ## not possible
    # # Not fitted imputation model with predict and score methods
    # with pytest.raises(ValueError):
    #     fitted_model = LinearRegression().fit(X, y)
    #     cpi(
    #         X,
    #         X,
    #         y,
    #         estimator=fitted_model,
    #         imputation_model=LinearRegression(),
    #         method="predict",
    #     )
    # with pytest.raises(ValueError):
    #     fitted_model = LinearRegression().fit(X, y)
    #     cpi(
    #         X,
    #         X,
    #         y,
    #         estimator=fitted_model,
    #         imputation_model=LinearRegression(),
    #         method="predict",
    #     )
