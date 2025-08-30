import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from hidimstat import PFI
from hidimstat._utils.scenario import multivariate_simulation


def test_permutation_importance():
    """Test the Permutation Importance algorithm on a linear scenario."""
    X, y, beta, noise = multivariate_simulation(
        n_samples=150,
        n_features=200,
        support_size=10,
        shuffle=False,
        seed=42,
    )
    important_features = np.where(beta != 0)[0]
    non_important_features = np.where(beta == 0)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    pfi = PFI(
        estimator=regression_model,
        n_permutations=20,
        method="predict",
        random_state=0,
        n_jobs=1,
    )

    pfi.fit(
        X_train,
        y_train,
        groups=None,
    )
    vim = pfi.importance(X_test, y_test)

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
    pfi = PFI(
        estimator=regression_model,
        n_permutations=20,
        method="predict",
        random_state=0,
        n_jobs=1,
    )
    pfi.fit(
        X_train_df,
        y_train,
        groups=groups,
    )
    # warnings because we doesn't considere the name of columns of pandas
    with pytest.warns(UserWarning, match="X does not have valid feature names, but"):
        vim = pfi.importance(X_test_df, y_test)

    importance = vim["importance"]
    assert importance[0].mean() > importance[1].mean()

    # Classification case
    y_clf = np.where(y > np.median(y), 1, 0)
    _, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, random_state=0)
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train_clf)

    pfi_clf = PFI(
        estimator=logistic_model,
        n_permutations=20,
        method="predict_proba",
        random_state=0,
        n_jobs=1,
        loss=log_loss,
    )

    pfi_clf.fit(
        X_train,
        y_train_clf,
        groups=None,
    )
    vim_clf = pfi_clf.importance(X_test, y_test_clf)

    importance_clf = vim_clf["importance"]
    assert importance_clf.shape == (X.shape[1],)


@pytest.fixture
def pfi_test_data():
    X, y, _, _ = multivariate_simulation(
        n_samples=100,
        n_features=5,
        support_size=2,
        rho=0,
        value=1,
        signal_noise_ratio=4,
        rho_serial=0,
        shuffle=False,
        seed=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, model


def test_pfi_multiple_calls_are_reproducible(pfi_test_data):
    X_train, X_test, y_train, y_test, model = pfi_test_data
    pfi = PFI(
        estimator=model,
        n_permutations=20,
        method="predict",
        random_state=0,
        n_jobs=1,
    )
    pfi.fit(X_train, y_train, groups=None)
    vim = pfi.importance(X_test, y_test)["importance"]
    vim_reproducible = pfi.importance(X_test, y_test)["importance"]
    assert np.array_equal(vim, vim_reproducible)


def test_pfi_different_instances_same_random_state_are_reproducible(pfi_test_data):
    X_train, X_test, y_train, y_test, model = pfi_test_data
    pfi_1 = PFI(
        estimator=model,
        n_permutations=20,
        method="predict",
        random_state=0,
        n_jobs=1,
    )
    pfi_1.fit(X_train, y_train, groups=None)
    vim_1 = pfi_1.importance(X_test, y_test)["importance"]

    pfi_2 = PFI(
        estimator=model,
        n_permutations=20,
        method="predict",
        random_state=0,
        n_jobs=1,
    )
    pfi_2.fit(X_train, y_train, groups=None)
    vim_2 = pfi_2.importance(X_test, y_test)["importance"]
    assert np.array_equal(vim_1, vim_2)


def test_pfi_different_random_state_is_not_reproducible(pfi_test_data):
    X_train, X_test, y_train, y_test, model = pfi_test_data
    pfi_fixed = PFI(
        estimator=model,
        n_permutations=20,
        method="predict",
        random_state=0,
        n_jobs=1,
    )
    pfi_fixed.fit(X_train, y_train, groups=None)
    vim_fixed = pfi_fixed.importance(X_test, y_test)["importance"]

    pfi_new_state = PFI(
        estimator=model,
        n_permutations=20,
        method="predict",
        random_state=1,
        n_jobs=1,
    )
    pfi_new_state.fit(X_train, y_train, groups=None)
    vim_new_state = pfi_new_state.importance(X_test, y_test)["importance"]
    assert not np.array_equal(vim_fixed, vim_new_state)

    pfi_none_state = PFI(
        estimator=model,
        n_permutations=20,
        method="predict",
        random_state=None,
        n_jobs=1,
    )
    pfi_none_state.fit(X_train, y_train, groups=None)
    vim_none_state_1 = pfi_none_state.importance(X_test, y_test)["importance"]
    vim_none_state_2 = pfi_none_state.importance(X_test, y_test)["importance"]
    assert not np.array_equal(vim_none_state_1, vim_none_state_2)
