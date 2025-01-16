import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from hidimstat.permutation_importance_scikitlearn import permutation_importance


def test_permutation_importance(linear_scenario):
    X, y, beta = linear_scenario
    important_features = np.where(beta != 0)[0]
    non_important_features = np.where(beta == 0)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    vim = permutation_importance(
        regression_model,
        X_test,
        y_test,
        n_repeats=20,
        scoring="r2",
        random_state=0,
        n_jobs=1,
    )

    importance = vim["importances_mean"]

    assert importance.shape == (X.shape[1],)
    assert (
        importance[important_features].mean()
        > importance[non_important_features].mean()
    )

    # impossible with groups
    # # Same with groups and a pd.DataFrame
    # groups = {
    #     "group_0": [f"col_{i}" for i in important_features],
    #     "the_group_1": [f"col_{i}" for i in non_important_features],
    # }
    # X_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
    # X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, random_state=0)
    # regression_model = LinearRegression()
    # regression_model.fit(X_train_df, y_train)
    # vim = permutation_importance(
    #     regression_model,
    #     X_test_df,
    #     y_test,
    #     n_repeats=20,
    #     scoring='r2',
    #     random_state=0,
    #     n_jobs=1,
    #     groups=groups
    # )
    # importance = vim['importances_mean']
    #
    # assert importance[0].mean() > importance[1].mean()

    # Classification case
    y_clf = np.where(y > np.median(y), 1, 0)
    _, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, random_state=0)
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train_clf)
    vim_clf = permutation_importance(
        logistic_model,
        X_test,
        y_test_clf,
        n_repeats=20,
        scoring="neg_log_loss",
        random_state=0,
        n_jobs=1,
    )
    importance_clf = vim_clf["importances_mean"]

    assert importance_clf.shape == (X.shape[1],)
