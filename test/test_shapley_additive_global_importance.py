import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from hidimstat import SAGE
from hidimstat._utils.scenario import multivariate_simulation

parameter_smoke = [
    ("sage", 150, 20, 5, 0.2, 42, 1.0, 4.0, 0.0),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_smoke, strict=False))[1:]), strict=False),
    ids=next(zip(*parameter_smoke, strict=False)),
)
def test_sage_smoke(data_generator):
    """Smoke test for SAGE. Checks that the importance values are computed and
    have the expected shape.
    """
    X, y, important_features, _ = data_generator

    model = LinearRegression()
    model.fit(X, y)
    sage = SAGE(
        estimator=model, n_permutations=10, n_subsets=20, random_state=42
    )
    sage.fit(X)
    importance = sage.importance(X, y)
    assert importance.shape == (X.shape[1],)
    assert (importance[important_features] > 0.0).all()


@pytest.fixture(scope="module")
def sage_test_data():
    """
    Fixture to generate test data and a fitted LinearRegression model for SAGE
    reproducibility tests.
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
    sage_default_parameters = {
        "estimator": model,
        "method": "predict",
        "n_subsets": 20,
        "n_permutations": 10,
        "n_jobs": 1,
    }
    return X_train, y_train, X_test, y_test, sage_default_parameters


def test_sage_randomness_with_none(sage_test_data):
    """
    Test that multiple calls of .importance() when SAGE has random_state=None
    produce different results.
    """
    X_train, y_train, X_test, y_test, sage_default_parameters = sage_test_data
    sage = SAGE(random_state=None, **sage_default_parameters)
    sage.fit(X_train, y_train)
    vim = sage.importance(X_test, y_test)

    # repeat importance
    vim_repeat = sage.importance(X_test, y_test)
    assert not np.array_equal(vim, vim_repeat)

    # repeat with a new SAGE instance
    sage_2 = SAGE(random_state=None, **sage_default_parameters)
    sage_2.fit(X_train, y_train)
    vim_reproducibility = sage_2.importance(X_test, y_test)
    assert not np.array_equal(vim, vim_reproducibility)


def test_sage_reproducibility_with_integer(sage_test_data):
    """
    Test that multiple calls of .importance() when SAGE has random_state=int
    produce identical results.
    """
    X_train, y_train, X_test, y_test, sage_default_parameters = sage_test_data
    sage = SAGE(random_state=0, **sage_default_parameters)
    sage.fit(X_train, y_train)
    vim = sage.importance(X_test, y_test)
    # repeat importance
    vim_repeat = sage.importance(X_test, y_test)
    assert np.array_equal(vim, vim_repeat)

    # refit (should not change anything for marginal sage)
    sage.fit(X_train)
    vim_refit = sage.importance(X_test, y_test)
    assert np.array_equal(vim, vim_refit)

    # Reproducibility
    sage_2 = SAGE(random_state=0, **sage_default_parameters)
    sage_2.fit(X_train, y_train)
    vim_reproducibility = sage_2.importance(X_test, y_test)
    assert np.array_equal(vim, vim_reproducibility)


def test_sage_reproducibility_with_rng(sage_test_data):
    """
    Test that:
     1. Multiple calls of .importance() when SAGE has random_state=rng are random
     2. refit with a fresh rng of the same seed provides the same result
    """
    X_train, y_train, X_test, y_test, sage_default_parameters = sage_test_data
    rng = np.random.default_rng(0)
    sage = SAGE(random_state=rng, **sage_default_parameters)
    sage.fit(X_train, y_train)
    vim = sage.importance(X_test, y_test)
    # repeat importance
    vim_repeat = sage.importance(X_test, y_test)
    assert not np.array_equal(vim, vim_repeat)

    # refit repeatability with a new rng of the same seed
    sage.random_state = np.random.default_rng(0)
    sage.fit(X_train, y_train)
    vim_refit = sage.importance(X_test, y_test)
    assert np.array_equal(vim, vim_refit)

    # Reproducibility
    sage_2 = SAGE(
        random_state=np.random.default_rng(0), **sage_default_parameters
    )
    sage_2.fit(X_train, y_train)
    vim_reproducibility = sage_2.importance(X_test, y_test)
    assert np.array_equal(vim, vim_reproducibility)
