import numpy as np
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import accuracy_score

from hidimstat.conditional_sampling import ConditionalSampler


def test_continuous_case():
    n = 1000
    sampler = ConditionalSampler(
        model=RidgeCV(alphas=np.logspace(-2, 2, 10)),
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
    n = 1000

    sampler = ConditionalSampler(
        model=LogisticRegressionCV(Cs=np.logspace(-2, 2, 10)),
        data_type="binary",
        random_state=0,
    )

    # Test for perfectly correlated features
    X = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=n)
    X[:, 1] = (X[:, 0] > 0).astype(float)
    sampler.fit(np.delete(X, 1, axis=1), X[:, 1])
    n_samples = 10
    X_1_perm = sampler.sample(np.delete(X, 1, axis=1), X[:, 1], n_samples=n_samples)
    for i in range(n_samples):
        assert accuracy_score(X_1_perm[i], X[:, 1]) > 0.8

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
        model_classification=LogisticRegressionCV(Cs=np.logspace(-2, 2, 10)),
        model_regression=RidgeCV(alphas=np.logspace(-2, 2, 10)),
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
