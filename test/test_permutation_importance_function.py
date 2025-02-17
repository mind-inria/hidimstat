import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from hidimstat.permutation_importance_func import permutation_importance


def test_permutation_importance_no_fitting(linear_scenario):
    X, y, beta = linear_scenario
    important_features = np.where(beta != 0)[0]
    non_important_features = np.where(beta == 0)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    regression_model = LinearRegression()
    importance, list_loss_j, loss_reference = permutation_importance(
        X_test,
        y_test,
        estimator=regression_model,
        n_permutations=20,
        method="predict",
        random_state=0,
        n_jobs=1,
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
    regression_model = LinearRegression()
    importance, list_loss_j, loss_reference = permutation_importance(
        X_test_df,
        y_test,
        estimator=regression_model,
        n_permutations=20,
        method="predict",
        random_state=0,
        n_jobs=1,
        groups=groups,
    )

    assert importance[0].mean() > importance[1].mean()

    # Same with groups
    groups = {
        "group_0": [i for i in important_features],
        "the_group_1": [i for i in non_important_features],
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    regression_model = LinearRegression()
    importance, list_loss_j, loss_reference = permutation_importance(
        X_test,
        y_test,
        estimator=regression_model,
        n_permutations=20,
        method="predict",
        random_state=0,
        n_jobs=1,
        groups=groups,
    )

    assert importance[0].mean() > importance[1].mean()

    # Classification case
    y_clf = np.zeros_like(y)
    for i, quantile in enumerate(np.arange(0.2, 0.8, 0.2)):
        y_clf[np.where(y > np.quantile(y, quantile))] = i
    _, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, random_state=0)
    logistic_model = LogisticRegression()

    importance_clf, list_loss_j, loss_reference = permutation_importance(
        X_test,
        y_test_clf,
        estimator=logistic_model,
        n_permutations=20,
        method="predict_proba",
        random_state=0,
        n_jobs=1,
        loss=log_loss,
        groups=None,
    )

    assert importance_clf.shape == (X.shape[1],)


def test_with_fitting(linear_scenario):
    X, y, beta = linear_scenario
    important_features = np.where(beta != 0)[0]
    non_important_features = np.where(beta == 0)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    importance, list_loss_j, loss_reference = permutation_importance(
        X_test,
        y_test,
        estimator=regression_model,
        n_permutations=20,
        method="predict",
        random_state=0,
        n_jobs=1,
    )

    assert importance.shape == (X.shape[1],)
    assert (
        importance[important_features].mean()
        > importance[non_important_features].mean()
    )
