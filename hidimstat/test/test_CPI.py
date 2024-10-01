import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from hidimstat.cpi import CPI


def test_CPI(linear_scenario):
    X, y, beta = linear_scenario
    important_features = np.where(beta != 0)[0]
    non_important_features = np.where(beta == 0)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    covariate_estimator = LinearRegression()

    cpi = CPI(
        estimator=regression_model,
        covariate_estimator=covariate_estimator,
        n_perm=20,
        groups=None,
        score_proba=False,
        random_state=0,
        n_jobs=1,
    )

    cpi.fit(X_train, y_train)
    vim = cpi.predict(X_test, y_test)

    importance = vim["importance"]
    assert importance.shape == (X.shape[1],)
    assert (
        importance[important_features].mean()
        > importance[non_important_features].mean()
    )

    # Same with groups
    groups = {0: important_features, 1: non_important_features}
    cpi = CPI(
        estimator=regression_model,
        covariate_estimator=covariate_estimator,
        n_perm=20,
        groups=groups,
        score_proba=False,
        random_state=0,
        n_jobs=1,
    )
    cpi.fit(X_train, y_train)
    vim = cpi.predict(X_test, y_test)

    importance = vim["importance"]
    assert importance[0].mean() > importance[1].mean()
