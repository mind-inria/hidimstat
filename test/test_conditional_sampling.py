import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (
    HuberRegressor,
    LogisticRegressionCV,
    RidgeClassifier,
    RidgeCV,
)
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from hidimstat.conditional_sampling import ConditionalSampler


def test_continuous_case():
    """Test sampling from the conditional distribution of a continuous variable."""
    n = 1000
    np.random.seed(40)
    sampler = ConditionalSampler(
        model_regression=RidgeCV(alphas=np.logspace(-2, 2, 10)),
        data_type="continuous",
        random_state=0,
    )

    # Test for perfectly correlated features
    X = np.random.randn(n, 2)
    X[:, 1] = X[:, 0]

    sampler.fit(np.delete(X, 1, axis=1), X[:, 1])
    n_samples = 10
    X_1_perm = sampler.sample(np.delete(X, 1, axis=1), X[:, 1], n_samples=n_samples)

    for i in range(n_samples):
        assert np.corrcoef(X_1_perm[i], X[:, 1])[0, 1] > 0.99

    # Test for uncorrelated features
    X = np.random.randn(n, 2)
    sampler.fit(np.delete(X, 1, axis=1), X[:, 1])
    n_samples = 10
    X_1_perm = sampler.sample(np.delete(X, 1, axis=1), X[:, 1], n_samples=n_samples)
    for i in range(n_samples):
        assert np.corrcoef([X_1_perm[i], X[:, 1]])[0, 1] < 0.1

    # Test for medium correlated features
    X = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.6], [0.6, 1]], size=n)
    sampler.fit(np.delete(X, 1, axis=1), X[:, 1])
    n_samples = 10
    X_1_perm = sampler.sample(np.delete(X, 1, axis=1), X[:, 1], n_samples=n_samples)
    for i in range(n_samples):
        assert 0.2 < np.corrcoef([X_1_perm[i], X[:, 1]])[0, 1] < 0.8


def test_binary_case():
    """Test sampling from the conditional distribution of a binary variable."""
    n = 1000
    np.random.seed(40)

    sampler = ConditionalSampler(
        model_categorical=LogisticRegressionCV(Cs=np.logspace(-2, 2, 10)),
        data_type="categorical",
        random_state=0,
    )

    # Test for perfectly correlated features
    X = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=n)
    X[:, 1] = (X[:, 0] > 0).astype(float)
    sampler.fit(np.delete(X, 1, axis=1), X[:, 1])
    n_samples = 10
    X_1_perm = sampler.sample(np.delete(X, 1, axis=1), X[:, 1], n_samples=n_samples)
    for i in range(n_samples):
        assert accuracy_score(X_1_perm[i], X[:, 1]) > 0.7

    # independent features
    X = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=n)
    X[:, 1] = np.random.randint(0, 2, size=n)
    sampler.fit(np.delete(X, 1, axis=1), X[:, 1])
    n_samples = 10
    X_1_perm = sampler.sample(np.delete(X, 1, axis=1), X[:, 1], n_samples=n_samples)
    for i in range(n_samples):
        # chance level at 0.5
        assert accuracy_score(X_1_perm[i], X[:, 1]) < 0.6

    sampler = ConditionalSampler(
        model_regression=RidgeCV(alphas=np.logspace(-2, 2, 10)),
        model_categorical=LogisticRegressionCV(Cs=np.logspace(-2, 2, 10)),
        data_type="auto",
        random_state=0,
    )
    X = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.0], [0.0, 1]], size=n)
    X[:, 1] = (X[:, 0] + np.random.randn(n) * -0.5) > 0
    sampler.fit(np.delete(X, 1, axis=1), X[:, 1])
    n_samples = 10
    X_1_perm = sampler.sample(np.delete(X, 1, axis=1), X[:, 1], n_samples=n_samples)
    for i in range(n_samples):
        assert 0.9 > accuracy_score(X_1_perm[i], X[:, 1]) > 0.6


def test_error_wrong_type_data():
    """Test for error when model does not have predict"""
    sampler = ConditionalSampler(
        data_type="wrong_type",
        random_state=0,
    )
    X = np.random.randint(0, 2, size=(100, 2))
    with pytest.raises(ValueError, match="type of data 'wrong_type' unknow."):
        sampler.fit(np.delete(X, 1, axis=1), X[:, 1])


def test_error_no_predic_proba():
    """Test for error when model does not have predict_proba"""
    np.random.seed(40)
    sampler = ConditionalSampler(
        model_categorical=RidgeClassifier(),
        data_type="categorical",
        random_state=0,
    )
    X = np.random.randint(0, 2, size=(100, 2))
    sampler.fit(np.delete(X, 1, axis=1), X[:, 1])
    with pytest.raises(
        AttributeError,
        match="The model must have a `predict_proba` method to be used for",
    ):
        sampler.sample(np.delete(X, 1, axis=1), X[:, 1])


def test_error_no_predic():
    """Test for error when model does not have predict"""
    np.random.seed(40)
    X = np.random.randint(0, 2, size=(100, 2))
    sampler = ConditionalSampler(
        model_regression=StandardScaler(),
        data_type="continuous",
        random_state=0,
    )
    sampler.fit(np.delete(X, 1, axis=1), X[:, 1])
    with pytest.raises(
        AttributeError, match="The model must have a `predict` method to be used for"
    ):
        sampler.sample(np.delete(X, 1, axis=1), X[:, 1])


def test_error_no_model_provide():
    """Test when there is no model for the category"""
    np.random.seed(40)
    X = np.random.randint(0, 2, size=(100, 2))
    sampler = ConditionalSampler(
        data_type="auto",
        random_state=0,
    )
    with pytest.raises(
        AttributeError, match="No model was provided for categorical data"
    ):
        sampler.fit(np.delete(X, 1, axis=1), X[:, 1])
    X = np.random.randn(100, 2)
    sampler = ConditionalSampler(
        data_type="auto",
        random_state=0,
    )
    with pytest.raises(
        AttributeError, match="No model was provided for continuous data"
    ):
        sampler.fit(np.delete(X, 1, axis=1), X[:, 1])


def test_group_case():
    """Test for group case: Sample a group of variables given the complementary
    set of variables."""
    np.random.seed(40)
    cov_matrix = np.ones((4, 4)) * 0.6
    np.fill_diagonal(cov_matrix, 1)
    n = 1000
    X = np.random.multivariate_normal(mean=np.zeros(4), cov=cov_matrix, size=n)
    # test with a model that DOES NOT nativly support multioutput
    sampler = ConditionalSampler(
        model_regression=RandomForestRegressor(
            n_estimators=10, max_depth=10, random_state=0
        ),
        data_type="continuous",
        random_state=0,
    )
    sampler.fit(X[:, :2], X[:, 2:])
    n_samples = 10
    X_2_perm = sampler.sample(X[:, :2], X[:, 2:], n_samples=n_samples)
    assert X_2_perm.shape == (n_samples, n, 2)
    for i in range(n_samples):
        assert 0.2 < np.corrcoef([X_2_perm[i, :, 0], X[:, 2]])[0, 1] < 0.9
        assert 0.2 < np.corrcoef([X_2_perm[i, :, 1], X[:, 3]])[0, 1] < 0.9
    # test with a model DOES nativly support multioutput
    sampler = ConditionalSampler(
        model_regression=HuberRegressor(),
        data_type="continuous",
        random_state=0,
    )
    sampler.fit(X[:, :2], X[:, 2:])

    # Binary case
    X = np.random.randn(n, 5)
    X[:, 3] = X[:, 0] + X[:, 1] + X[:, 2] + np.random.randn(X.shape[0]) * 0.3 > 0
    X[:, 4] = 2 * X[:, 1] - 1 + np.random.randn(X.shape[0]) * 0.3 > 0
    model = LogisticRegressionCV(Cs=np.logspace(-2, 2, 10))
    sampler = ConditionalSampler(
        model_categorical=model,
        data_type="categorical",
        random_state=0,
    )
    sampler.fit(X[:, :3], X[:, 3:])

    n_samples = 10
    X_3_perm = sampler.sample(X[:, :3], X[:, 3:], n_samples=n_samples)
    assert X_3_perm.shape == (n_samples, X.shape[0], 2)
    for i in range(n_samples):
        # TODO check why so good accuracy
        assert 0.96 > accuracy_score(X_3_perm[i, :, 0], X[:, 3]) > 0.7
        assert 0.96 > accuracy_score(X_3_perm[i, :, 1], X[:, 4]) > 0.7


def test_sample_categorical():
    """Test for categorical case with both single and groups of variables."""
    np.random.seed(40)
    n = 1000
    X = np.random.randn(n, 5)
    X[:, 3] = np.digitize(
        X[:, 0], bins=np.quantile(X[:, 0], [0, 0.2, 0.4, 0.6, 0.8, 1]), right=True
    )
    X[:, 4] = np.digitize(
        X[:, 1], bins=np.quantile(X[:, 1], [0, 0.2, 0.4, 0.6, 0.8, 1]), right=True
    )
    X[:, 3:][np.where(X[:, 3:] == 0)] = 1

    sampler = ConditionalSampler(
        model_categorical=LogisticRegressionCV(Cs=np.logspace(-2, 2, 10)),
        data_type="categorical",
        random_state=0,
    )

    sampler.fit(X[:, :3], X[:, 3:])
    n_samples = 10
    X_3_perm = sampler.sample(X[:, :3], X[:, 3:], n_samples=n_samples)
    assert X_3_perm.shape == (n_samples, X.shape[0], 2)
    for i in range(n_samples):
        # Chance level is now 1/5
        assert 0.991 > accuracy_score(X_3_perm[i, :, 0], X[:, 3]) > 0.3
        assert 0.991 > accuracy_score(X_3_perm[i, :, 1], X[:, 4]) > 0.3

    # Using a provided OneVsRestClassifier
    sampler = ConditionalSampler(
        model_categorical=OneVsRestClassifier(
            LogisticRegressionCV(Cs=np.logspace(-2, 2, 10))
        ),
        data_type="categorical",
        random_state=0,
    )

    sampler.fit(X[:, :3], X[:, 3:])
    n_samples = 10
    X_3_perm = sampler.sample(X[:, :3], X[:, 3:], n_samples=n_samples)
    assert X_3_perm.shape == (n_samples, X.shape[0], 2)
    for i in range(n_samples):
        # Chance level is now 1/5
        assert 0.95 > accuracy_score(X_3_perm[i, :, 0], X[:, 3]) > 0.3
        assert 0.95 > accuracy_score(X_3_perm[i, :, 1], X[:, 4]) > 0.3
        assert 0.95 > accuracy_score(X_3_perm[i, :, 1], X[:, 4]) > 0.3
