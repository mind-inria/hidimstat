import pytest
from sklearn.linear_model import LinearRegression

from hidimstat import SAGE

parameter_smoke = [
    ("sage", 150, 20, 5, 0.2, 42, 1.0, 4.0, 0.0),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_smoke, strict=False))[1:]), strict=False),
    ids=next(zip(*parameter_smoke, strict=False)),
)
def test_sage_smoke(data_generator):
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
