import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from hidimstat.loco import LOCO


def test_LOCO(linear_scenario):
    X, y, beta = linear_scenario
    important_features = np.where(beta != 0)[0]
    non_important_features = np.where(beta == 0)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    loco = LOCO(
        estimator=regression_model,
        score_proba=False,
        random_state=0,
        n_jobs=1,
    )

    loco.fit(
        X_train,
        y_train,
        groups=None,
    )
    vim = loco.score(X_test, y_test)

    importance = vim["importance"]
    assert importance.shape == (X.shape[1],)
    assert (
        importance[important_features].mean()
        > importance[non_important_features].mean()
    )

    # Same with groups
    groups = {0: important_features, 1: non_important_features}
    loco = LOCO(
        estimator=regression_model,
        score_proba=False,
        random_state=0,
        n_jobs=1,
    )
    loco.fit(
        X_train,
        y_train,
        groups=groups,
    )
    vim = loco.score(X_test, y_test)

    importance = vim["importance"]
    assert importance[0].mean() > importance[1].mean()
