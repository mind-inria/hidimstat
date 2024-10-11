import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from hidimstat.permutation_importance import PermutationImportance


def test_CPI(linear_scenario):
    X, y, beta = linear_scenario
    important_features = np.where(beta != 0)[0]
    non_important_features = np.where(beta == 0)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    pi = PermutationImportance(
        estimator=regression_model,
        n_perm=20,
        score_proba=False,
        random_state=0,
        n_jobs=1,
    )

    pi.fit(
        X_train,
        y_train,
        groups=None,
    )
    vim = pi.score(X_test, y_test)

    importance = vim["importance"]
    assert importance.shape == (X.shape[1],)
    assert (
        importance[important_features].mean()
        > importance[non_important_features].mean()
    )

    # Same with groups
    groups = {0: important_features, 1: non_important_features}
    pi = PermutationImportance(
        estimator=regression_model,
        n_perm=20,
        score_proba=False,
        random_state=0,
        n_jobs=1,
    )
    pi.fit(
        X_train,
        y_train,
        groups=groups,
    )
    vim = pi.score(X_test, y_test)

    importance = vim["importance"]
    assert importance[0].mean() > importance[1].mean()
