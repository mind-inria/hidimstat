import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from hidimstat.cpi import CPI


def test_cpi(linear_scenario):
    X, y, beta = linear_scenario
    important_features = np.where(beta != 0)[0]
    non_important_features = np.where(beta == 0)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    imputation_model = LinearRegression()

    cpi = CPI(
        estimator=regression_model,
        imputation_model=imputation_model,
        n_permutations=20,
        score_proba=False,
        random_state=0,
        n_jobs=1,
    )

    cpi.fit(
        X_train,
        y_train,
        groups=None,
    )
    vim = cpi.score(X_test, y_test)

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
    imputation_model_list = [clone(imputation_model) for _ in range(2)]
    regression_model.fit(X_train_df, y_train)
    cpi = CPI(
        estimator=regression_model,
        imputation_model=imputation_model_list,
        n_permutations=20,
        score_proba=False,
        random_state=0,
        n_jobs=1,
    )
    cpi.fit(
        X_train_df,
        y_train,
        groups=groups,
    )
    vim = cpi.score(X_test_df, y_test)

    importance = vim["importance"]
    assert importance[0].mean() > importance[1].mean()

    # Classification scenario
    y_clf = np.where(y > 0, 1, 0)

    _, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, random_state=0)
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train_clf)

    cpi = CPI(
        estimator=logistic_model,
        imputation_model=imputation_model,
        n_permutations=20,
        random_state=0,
        n_jobs=1,
        score_proba=True,
        loss=log_loss,
    )
    cpi.fit(
        X_train,
        y=None,
        groups=None,
    )
    vim = cpi.score(X_test, y_test_clf)
