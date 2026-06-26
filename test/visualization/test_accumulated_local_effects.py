import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from hidimstat._utils.scenario import multivariate_simulation
from hidimstat.visualization import ALE
from hidimstat.visualization.accumulated_local_effects import (
    _bin_indices,
    _build_quantile_grid,
    _predict_fn,
    compute_ale_1d_continuous,
    compute_ale_1d_discrete,
    compute_ale_2d,
)

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def ale_test_data():
    """Fixture generating both continuous and discrete data configurations for ALE tests."""
    # Continuous
    X_continuous, y, beta, _ = multivariate_simulation(
        n_samples=300,
        n_features=20,
        support_size=3,
        rho=0.3,
        value=1.0,
        signal_noise_ratio=10.0,
        rho_serial=0.0,
        shuffle=False,
        seed=92,
    )
    important_features = np.where(beta != 0)[0]
    model_continous = LinearRegression()
    model_continous.fit(X_continuous, y)

    # Discrete
    X_discrete = X_continuous.copy()
    X_discrete[:, 0] = np.random.default_rng(92).choice(
        [0, 1, 2], size=X_continuous.shape[0]
    )
    X_discrete[0, 0] = 0
    X_discrete[1, 0] = 1

    model_discrete = LinearRegression()
    model_discrete.fit(X_discrete, y)

    # Categorical
    X_categorical = X_continuous.astype(object).copy()
    X_categorical[:, 0] = "A"

    model_categorical = HistGradientBoostingRegressor(categorical_features=[0])
    model_categorical.fit(X_categorical, y)

    return {
        "continuous": {
            "X": X_continuous,
            "y": y,
            "model": model_continous,
            "important_features": important_features,
        },
        "discrete": {
            "X": X_discrete,
            "y": y,
            "model": model_discrete,
        },
        "categorical": {
            "X": X_categorical,
            "y": y,
            "model": model_categorical,
        },
    }


def test_predict_fn():
    """Test all branches of _predict_fn."""

    class BaseDummy(BaseEstimator):
        def __init__(self):
            super().__init__()
            self.fitted_ = True

        def fit(self, X, y=None):
            del X, y
            return self

    class DummyPredict(BaseDummy):
        def predict(self, X):
            del X
            return np.array([1.0, 2.0])

    class DummyPredictProbaBinary(BaseDummy):
        def predict_proba(self, X):
            del X
            return np.array([[0.1, 0.9], [0.2, 0.8]])

    class DummyPredictProbaMulti(BaseDummy):
        def predict_proba(self, X):
            del X
            return np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])

    class DummyDecisionFunction(BaseDummy):
        def decision_function(self, X):
            del X
            return np.array([0.5, -0.5])

    class DummyNoPredict(BaseDummy):
        pass

    class DummyWithFeatureNames(BaseDummy):
        def __init__(self):
            super().__init__()
            self.feature_names_in_ = ["A", "B"]

        def predict(self, X):
            return np.array([0] * len(X))

    X_dummy = np.array([[1, 2], [3, 4]])
    model = LinearRegression()

    with pytest.raises(NotFittedError):
        _predict_fn(model, X_dummy)

    np.testing.assert_array_equal(
        _predict_fn(DummyPredict(), X_dummy), [1.0, 2.0]
    )
    np.testing.assert_array_equal(
        _predict_fn(DummyPredictProbaBinary(), X_dummy), [0.9, 0.8]
    )
    with pytest.raises(
        ValueError, match="Multiclass models are not supported"
    ):
        _predict_fn(DummyPredictProbaMulti(), X_dummy)
    np.testing.assert_array_equal(
        _predict_fn(DummyDecisionFunction(), X_dummy), [0.5, -0.5]
    )
    with pytest.raises(
        ValueError, match="'estimator' must expose at least one"
    ):
        _predict_fn(DummyNoPredict(), X_dummy)

    res_names = _predict_fn(DummyWithFeatureNames(), X_dummy)
    np.testing.assert_array_equal(res_names, [0, 0])


def test_build_quantile_grid():
    """Test quantile grid creation."""
    x = np.arange(100)

    grid_auto = _build_quantile_grid(x, "auto")
    assert len(grid_auto) > 1

    with pytest.raises(
        ValueError, match="'percentiles' must be a tuple of 2 floats"
    ):
        _build_quantile_grid(x, 0, percentiles=[0.05, 0.95])
    with pytest.raises(
        ValueError, match="'percentiles' must be a tuple of 2 floats"
    ):
        _build_quantile_grid(x, 0, percentiles=(0.05, 0.25, 0.5, 0.75, 0.95))
    with pytest.raises(
        ValueError, match="'percentiles' must be a tuple of 2 floats"
    ):
        _build_quantile_grid(x, 0, percentiles=(-1, 0.5))
    with pytest.raises(
        ValueError, match="'percentiles' must be a tuple of 2 floats"
    ):
        _build_quantile_grid(x, 0, percentiles=(0.5, -1))

    with pytest.raises(
        ValueError,
        match="'grid_resolution' must be an int strictly greater than 0",
    ):
        _build_quantile_grid(x, "a")
    with pytest.raises(
        ValueError,
        match="'grid_resolution' must be an int strictly greater than 0",
    ):
        _build_quantile_grid(x, 0)
    with pytest.raises(
        ValueError,
        match="'grid_resolution' must be an int strictly greater than 0",
    ):
        _build_quantile_grid(x, -5)

    x_few = np.array([1, 1, 2, 2, 3])
    grid_few = _build_quantile_grid(
        x_few, grid_resolution=10, percentiles=(0, 1)
    )
    np.testing.assert_array_equal(grid_few, [1, 2, 3])


def test_bin_indices():
    """Verify that the bin assignment correctly handles extreme values ​​via clip."""
    quantiles = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])
    indices = _bin_indices(x, quantiles)
    np.testing.assert_array_equal(indices, [0, 0, 1, 2, 2])


def test_compute_ale_1d_continuous(ale_test_data):
    """Test the continuous 1D ALE calculation with and without confidence intervals."""
    data = ale_test_data["continuous"]
    X, model, important_features = (
        data["X"],
        data["model"],
        data["important_features"],
    )

    grid_resolution = 10

    # Without confidence intervals
    result = compute_ale_1d_continuous(
        model,
        X,
        feature_idx=important_features[0],
        grid_resolution=grid_resolution,
        confidence_interval=False,
    )
    assert "ale" in result
    assert "quantiles" in result
    assert "ale_err" in result
    assert isinstance(result["ale"], np.ndarray)
    assert isinstance(result["quantiles"], np.ndarray)
    assert result["ale_err"] is None
    assert len(result["ale"]) == len(result["quantiles"])
    assert len(result["quantiles"]) <= grid_resolution + 1

    # With confidence interval
    result_ci = compute_ale_1d_continuous(
        model,
        X,
        feature_idx=important_features[0],
        grid_resolution=grid_resolution,
        confidence_interval=True,
        confidence_level=0.95,
    )
    assert "ale" in result_ci
    assert "quantiles" in result_ci
    assert "ale_err" in result_ci
    assert isinstance(result_ci["ale"], np.ndarray)
    assert isinstance(result_ci["quantiles"], np.ndarray)
    assert isinstance(result_ci["ale_err"], np.ndarray)
    assert len(result_ci["ale"]) == len(result_ci["quantiles"])
    assert len(result_ci["quantiles"]) <= grid_resolution + 1
    assert len(result_ci["ale_err"]) == len(result_ci["quantiles"])


def test_compute_ale_1d_continuous_error(ale_test_data):
    """Check the raised error if the continuous feature does not have enough unique quantiles."""
    data = ale_test_data["continuous"]
    X, model = data["X"], data["model"]

    X_const = X.copy()
    X_const[:, 0] = 7

    with pytest.raises(
        ValueError, match="has fewer than 2 unique quantile edges"
    ):
        compute_ale_1d_continuous(
            model, X_const, feature_idx=0, grid_resolution=10
        )


def test_compute_ale_1d_discrete(ale_test_data):
    """Test the discrete 1D ALE calculation with and without confidence intervals."""
    data = ale_test_data["discrete"]
    X, model = data["X"], data["model"]

    # Without confidence intervals
    result = compute_ale_1d_discrete(
        model, X, feature_idx=0, confidence_interval=False
    )
    assert "ale" in result
    assert "unique_values" in result
    assert "ale_err" in result
    assert isinstance(result["ale"], np.ndarray)
    assert isinstance(result["unique_values"], np.ndarray)
    assert result["ale_err"] is None
    assert len(result["ale"]) == len(result["unique_values"])
    assert len(result["unique_values"]) <= 3

    # With confidence intervals
    result_ci = compute_ale_1d_discrete(
        model,
        X,
        feature_idx=0,
        confidence_interval=True,
        confidence_level=0.90,
    )
    assert "ale" in result_ci
    assert "unique_values" in result_ci
    assert "ale_err" in result_ci
    assert isinstance(result_ci["ale"], np.ndarray)
    assert isinstance(result_ci["unique_values"], np.ndarray)
    assert isinstance(result_ci["ale_err"], np.ndarray)
    assert len(result_ci["ale"]) == len(result_ci["unique_values"])
    assert len(result_ci["unique_values"]) <= 3
    assert len(result_ci["ale_err"]) == len(result_ci["unique_values"])


def test_compute_ale_1d_discrete_error(ale_test_data):
    """Check the raised error if the discrete feature does not have enough unique values."""
    data = ale_test_data["discrete"]
    X, model = data["X"], data["model"]

    X_const = X.copy()
    X_const[:, 0] = 7

    with pytest.raises(ValueError, match="has fewer than 2 unique values"):
        compute_ale_1d_discrete(model, X_const, feature_idx=0)

    with pytest.raises(
        ValueError, match="'percentiles' must be a tuple of 2 floats"
    ):
        compute_ale_1d_discrete(
            model, X, feature_idx=0, percentiles=[0.05, 0.95]
        )
    with pytest.raises(
        ValueError, match="'percentiles' must be a tuple of 2 floats"
    ):
        compute_ale_1d_discrete(
            model,
            X,
            feature_idx=0,
            percentiles=(0.05, 0.25, 0.5, 0.75, 0.95),
        )
    with pytest.raises(
        ValueError, match="'percentiles' must be a tuple of 2 floats"
    ):
        compute_ale_1d_discrete(model, X, feature_idx=0, percentiles=(-1, 0.5))
    with pytest.raises(
        ValueError, match="'percentiles' must be a tuple of 2 floats"
    ):
        compute_ale_1d_discrete(model, X, feature_idx=0, percentiles=(0.5, -1))


def test_compute_ale_2d(ale_test_data):
    """Test the 2D ALE calculation"""
    data = ale_test_data["continuous"]
    X, model, important_features = (
        data["X"],
        data["model"],
        data["important_features"],
    )

    grid_resolution = 5

    result = compute_ale_2d(
        model,
        X,
        feature_indices=[important_features[0], important_features[1]],
        grid_resolution=grid_resolution,
    )

    assert "ale" in result
    assert "quantiles_i" in result
    assert "quantiles_j" in result
    assert isinstance(result["ale"], np.ndarray)
    assert isinstance(result["quantiles_i"], np.ndarray)
    assert isinstance(result["quantiles_j"], np.ndarray)
    assert result["ale"].shape == (
        len(result["quantiles_i"]),
        len(result["quantiles_j"]),
    )
    assert len(result["quantiles_i"]) <= grid_resolution + 1
    assert len(result["quantiles_j"]) <= grid_resolution + 1


def test_compute_ale_2d_errors(ale_test_data):
    """Check the raised error of the 2D ALE."""
    data = ale_test_data["continuous"]
    X, model, important_features = (
        data["X"],
        data["model"],
        data["important_features"],
    )

    with pytest.raises(
        ValueError, match="must contain exactly two feature indices"
    ):
        compute_ale_2d(model, X, feature_indices=[0])
    with pytest.raises(
        ValueError, match="must contain exactly two feature indices"
    ):
        compute_ale_2d(model, X, feature_indices=[0, 1, 2])

    X_const_i = X.copy()
    X_const_i[:, important_features[0]] = 1
    with pytest.raises(
        ValueError, match="has fewer than 2 unique quantile edges"
    ):
        compute_ale_2d(
            model,
            X_const_i,
            feature_indices=[important_features[0], important_features[1]],
        )

    X_const_j = X.copy()
    X_const_j[:, important_features[1]] = 2
    with pytest.raises(
        ValueError, match="has fewer than 2 unique quantile edges"
    ):
        compute_ale_2d(
            model,
            X_const_j,
            feature_indices=[important_features[0], important_features[1]],
        )


def test_ale_no_feature_names(ale_test_data):
    """Test PDP plotting default feature names generation."""
    data = ale_test_data["continuous"]
    X, model = data["X"], data["model"]

    ale = ALE(model)

    axes = ale.plot(X, features=0)
    assert axes is not None
    plt.close("all")


def test_ale_plot_smoke(ale_test_data):
    """ALE plot smoke test."""
    data_continuous = ale_test_data["continuous"]
    data_discrete = ale_test_data["discrete"]
    data_categorical = ale_test_data["categorical"]

    feature_names = [f"feat_{i}" for i in range(data_continuous["X"].shape[1])]

    ale_continuous = ALE(data_continuous["model"], feature_names=feature_names)
    ale_discrete = ALE(data_discrete["model"])
    ale_categorical = ALE(data_categorical["model"])

    ax_1d_continuous = ale_continuous.plot(
        data_continuous["X"],
        features=data_continuous["important_features"][0],
        grid_resolution=10,
        confidence_interval=True,
    )
    assert ax_1d_continuous is not None
    plt.close("all")

    ax_1d_discrete = ale_discrete.plot(
        data_discrete["X"], features=0, confidence_interval=True
    )
    assert ax_1d_discrete is not None
    plt.close("all")

    ax_2d = ale_continuous.plot(
        data_continuous["X"],
        features=[
            data_continuous["important_features"][0],
            data_continuous["important_features"][1],
        ],
        grid_resolution=5,
    )
    assert ax_2d is not None
    plt.close("all")

    with pytest.raises(ValueError, match="ALE plots are supported"):
        ale_continuous.plot(data_continuous["X"], features=[1, 2, 3])
    with pytest.raises(
        TypeError, match="'features' must be an int or a list of int"
    ):
        ale_continuous.plot(
            data_continuous["X"], features="invalid_feature_format"
        )
    with pytest.raises(
        ValueError, match="not supported for non numeric categorical features"
    ):
        ale_continuous.plot(
            data_categorical["X"],
            features=0,
            feature_type="categorical",
        )
