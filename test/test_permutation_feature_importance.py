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
        feature_groups=None,
        random_state=0,
        n_jobs=1,
    )

    pfi.fit(
        X_train,
        y_train,
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
        feature_groups=groups,
        random_state=0,
        n_jobs=1,
    )
    pfi.fit(
        X_train_df,
        y_train,
    )
    # warnings because we doesn't consider the name of columns of pandas
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
        loss=log_loss,
        feature_groups=None,
        random_state=0,
        n_jobs=1,
    )

    pfi_clf.fit(
        X_train,
        y_train_clf,
    )
    vim_clf = pfi_clf.importance(X_test, y_test_clf)

    importance_clf = vim_clf["importance"]
    assert importance_clf.shape == (X.shape[1],)


@pytest.fixture(scope="module")
def pfi_test_data():
    """
    Fixture to generate data and fitted model for PFI tests.
    """
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
    pfi_default_parameters = {
        "estimator": model,
        "n_permutations": 20,
        "method": "predict",
        "n_jobs": 1,
    }
    return X_train, X_test, y_train, y_test, pfi_default_parameters


def test_pfi_repeatability(pfi_test_data):
    """
    Test that multiple calls of .importance() when PFI is seeded provide deterministic
    results.
    """
    X_train, X_test, y_train, y_test, pfi_default_parameters = pfi_test_data
    pfi = PFI(**pfi_default_parameters, random_state=0)
    pfi.fit(X_train, y_train)
    vim = pfi.importance(X_test, y_test)["importance"]
    vim_reproducible = pfi.importance(X_test, y_test)["importance"]
    assert np.array_equal(vim, vim_reproducible)


def test_pfi_randomness_with_none(pfi_test_data):
    """
    Test that different random states provide different results and multiple calls
    of .importance() when random_state is None provide different results.
    """
    X_train, X_test, y_train, y_test, pfi_default_parameters = pfi_test_data
    pfi_fixed = PFI(**pfi_default_parameters, random_state=0)
    pfi_fixed.fit(X_train, y_train)
    vim_fixed = pfi_fixed.importance(X_test, y_test)["importance"]

    pfi_new_state = PFI(**pfi_default_parameters, random_state=1)
    pfi_new_state.fit(X_train, y_train)
    vim_new_state = pfi_new_state.importance(X_test, y_test)["importance"]
    assert not np.array_equal(vim_fixed, vim_new_state)

    pfi_none_state = PFI(**pfi_default_parameters, random_state=None)
    pfi_none_state.fit(X_train, y_train)
    vim_none_state_1 = pfi_none_state.importance(X_test, y_test)["importance"]
    vim_none_state_2 = pfi_none_state.importance(X_test, y_test)["importance"]
    assert not np.array_equal(vim_none_state_1, vim_none_state_2)


def test_pfi_reproducibility_with_integer(pfi_test_data):
    """
    Test that different instances of PFI with the same random state provide
    deterministic results.
    """
    X_train, X_test, y_train, y_test, pfi_default_parameters = pfi_test_data
    pfi_1 = PFI(**pfi_default_parameters, random_state=0)
    pfi_1.fit(X_train, y_train)
    vim_1 = pfi_1.importance(X_test, y_test)["importance"]

    pfi_2 = PFI(**pfi_default_parameters, random_state=0)
    pfi_2.fit(X_train, y_train)
    vim_2 = pfi_2.importance(X_test, y_test)["importance"]
    assert np.array_equal(vim_1, vim_2)


def test_pfi_reproducibility_with_rng(pfi_test_data):
    """
    Test that:
     1. Mmultiple calls of .importance() when PFI has random_state=rng are random
     2. refit with same rng provides same result
    """
    X_train, X_test, y_train, y_test, pfi_default_parameters = pfi_test_data
    rng = np.random.default_rng(0)
    pfi = PFI(**pfi_default_parameters, random_state=rng)
    pfi.fit(X_train, y_train)
    vim = pfi.importance(X_test, y_test)["importance"]
    vim_repeat = pfi.importance(X_test, y_test)["importance"]
    assert not np.array_equal(vim, vim_repeat)

    # Refit with same rng
    rng = np.random.default_rng(0)
    pfi_reproducibility = PFI(**pfi_default_parameters, random_state=rng)
    pfi_reproducibility.fit(X_train, y_train)
    vim_reproducibility = pfi_reproducibility.importance(X_test, y_test)["importance"]
    assert np.array_equal(vim, vim_reproducibility)
