import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from hidimstat.loco import LOCO


def test_loco(linear_scenario):
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

    # Classification case
    y_clf = np.where(y > np.median(y), 1, 0)
    _, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, random_state=0)
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train_clf)

    loco_clf = LOCO(
        estimator=logistic_model,
        score_proba=True,
        random_state=0,
        n_jobs=1,
        loss=log_loss,
    )
    loco_clf.fit(
        X_train,
        y_train_clf,
        groups=None,
    )
    vim_clf = loco_clf.score(X_test, y_test_clf)

    importance_clf = vim_clf["importance"]
    assert importance_clf.shape == (X.shape[1],)
