import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

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


def test_predict_fn():
    """Test all branches of _predict_fn."""

    class DummyPredict:
        def predict(self, X):
            del X
            return np.array([1.0, 2.0])

    class DummyPredictProbaBinary:
        def predict_proba(self, X):
            del X
            return np.array([[0.1, 0.9], [0.2, 0.8]])

    class DummyDecisionFunction:
        def decision_function(self, X):
            del X
            return np.array([0.5, -0.5])

    class DummyNoPredict:
        pass

    class DummyWithFeatureNames:
        def __init__(self):
            self.feature_names_in_ = ["A", "B"]

        def predict(self, X):
            return np.array([0] * len(X))

    X_dummy = np.array([[1, 2], [3, 4]])

    np.testing.assert_array_equal(
        _predict_fn(DummyPredict(), X_dummy), [1.0, 2.0]
    )
    np.testing.assert_array_equal(
        _predict_fn(DummyPredictProbaBinary(), X_dummy), [0.9, 0.8]
    )
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
        ValueError, match="'grid_resolution' must be strictly greater than 0"
    ):
        _build_quantile_grid(x, 0)
    with pytest.raises(
        ValueError, match="'grid_resolution' must be strictly greater than 0"
    ):
        _build_quantile_grid(x, -5)

    x_few = np.array([1, 1, 2, 2, 3])
    grid_few = _build_quantile_grid(x_few, 10)
    np.testing.assert_array_equal(grid_few, [1, 2, 3])


def test_bin_indices():
    """Verify that the bin assignment correctly handles extreme values ​​via clip."""
    quantiles = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])
    indices = _bin_indices(x, quantiles)
    np.testing.assert_array_equal(indices, [0, 0, 1, 2, 2])


parameters_ale = [
    ("ale_instance_1", 300, 20, 3, 0.3, 92, 1.0, 10.0, 0.0),
    ("ale_instance_2", 150, 10, 2, 0.5, 42, 2.0, 5.0, 0.1),
]

ale_ids = [p[0] for p in parameters_ale]
ale_values = [p[1:] for p in parameters_ale]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    ale_values,
    ids=ale_ids,
)
class TestALE:
    def test_compute_ale_1d_continuous(self, data_generator):
        """Test the continuous 1D ALE calculation with and without confidence intervals."""
        X, y, important_features, _ = data_generator

        model = LinearRegression()
        model.fit(X, y)

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

    def test_compute_ale_1d_continuous_error(self, data_generator):
        """Check the raised error if the continuous feature does not have enough unique quantiles."""
        X, y, _, _ = data_generator

        model = LinearRegression()
        model.fit(X, y)

        X_const = X.copy()
        X_const[:, 0] = 7

        with pytest.raises(
            ValueError, match="has fewer than 2 unique quantile edges"
        ):
            compute_ale_1d_continuous(
                model, X_const, feature_idx=0, grid_resolution=10
            )

    def test_compute_ale_1d_discrete(self, data_generator):
        """Test the discrete 1D ALE calculation with and without confidence intervals."""
        X, y, _, _ = data_generator

        X_discrete = X.copy()
        X_discrete[:, 0] = np.random.default_rng(92).choice(
            [0, 1, 2], size=X.shape[0]
        )

        model = LinearRegression()
        model.fit(X_discrete, y)

        # Without confidence intervals
        result = compute_ale_1d_discrete(
            model, X_discrete, feature_idx=0, confidence_interval=False
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
            X_discrete,
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

    def test_compute_ale_1d_discrete_error(self, data_generator):
        """Check the raised error if the discrete feature does not have enough unique values."""
        X, y, _, _ = data_generator

        model = LinearRegression()
        model.fit(X, y)

        X_const = X.copy()
        X_const[:, 0] = 7

        with pytest.raises(ValueError, match="has fewer than 2 unique values"):
            compute_ale_1d_discrete(model, X_const, feature_idx=0)

    def test_compute_ale_2d(self, data_generator):
        """Test the 2D ALE calculation"""
        X, y, important_features, _ = data_generator

        model = LinearRegression()
        model.fit(X, y)

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

    def test_compute_ale_2d_errors(self, data_generator):
        """Check the raised error of the 2D ALE."""
        X, y, _, _ = data_generator

        model = LinearRegression()
        model.fit(X, y)

        with pytest.raises(
            NotImplementedError,
            match="Categorical features are not yet supported",
        ):
            compute_ale_2d(
                model, X, feature_indices=[0, 1], is_categorical=(True, False)
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
        X_const_i[:, 0] = 1
        with pytest.raises(
            ValueError, match="has fewer than 2 unique quantile edges"
        ):
            compute_ale_2d(model, X_const_i, feature_indices=[0, 1])

        X_const_j = X.copy()
        X_const_j[:, 1] = 2
        with pytest.raises(
            ValueError, match="has fewer than 2 unique quantile edges"
        ):
            compute_ale_2d(model, X_const_j, feature_indices=[0, 1])

    def test_ale_resolve_feature_type(self, data_generator):
        """Test the inference type."""
        X, y, _, _ = data_generator

        X_discrete = X.copy()
        X_discrete[:, 0] = np.random.default_rng(92).choice(
            [0, 1, 2], size=X.shape[0]
        )

        X_str = np.array([["a"], ["b"], ["c"]], dtype=object)

        model = LinearRegression()
        model.fit(X_discrete, y)

        ale = ALE(model)

        assert (
            ale._resolve_feature_type(X, feature=0, feature_type="auto")
            == "continuous"
        )
        assert (
            ale._resolve_feature_type(
                X_discrete, feature=0, feature_type="auto"
            )
            == "discrete"
        )
        assert (
            ale._resolve_feature_type(X_str, feature=0, feature_type="auto")
            == "categorical"
        )

        assert (
            ale._resolve_feature_type(X, feature=0, feature_type="continuous")
            == "continuous"
        )
        assert (
            ale._resolve_feature_type(X, feature=0, feature_type="discrete")
            == "discrete"
        )
        assert (
            ale._resolve_feature_type(X, feature=0, feature_type="categorical")
            == "categorical"
        )

        with pytest.raises(
            ValueError, match="feature_type should be a string among"
        ):
            ale._resolve_feature_type(X, 0, "invalid_type")

    def test_ale_no_feature_names(self, data_generator):
        """Test PDP plotting default feature names generation."""
        X, y, _, _ = data_generator

        model = LinearRegression()
        model.fit(X, y)

        ale = ALE(model)

        axes = ale.plot(X, features=0)
        assert axes is not None
        plt.close("all")

    def test_ale_plot_smoke(self, data_generator):
        """ALE plot smoke test."""
        X, y, _, _ = data_generator

        X_discrete = X.copy()
        X_discrete[:, 0] = np.random.default_rng(92).choice(
            [0, 1, 2], size=X.shape[0]
        )

        model = LinearRegression()
        model.fit(X_discrete, y)

        feature_names = [f"feat_{i}" for i in range(X_discrete.shape[1])]
        ale = ALE(model, feature_names=feature_names)

        ax_1d_continuous = ale.plot(
            X_discrete,
            features=1,
            grid_resolution=10,
            confidence_interval=True,
        )
        assert ax_1d_continuous is not None
        plt.close("all")

        ax_1d_discrete = ale.plot(
            X_discrete, features=0, confidence_interval=True
        )
        assert ax_1d_discrete is not None
        plt.close("all")

        ax_2d = ale.plot(X_discrete, features=[1, 2], grid_resolution=5)
        assert ax_2d is not None
        plt.close("all")

        with pytest.raises(ValueError, match="ALE plots are supported"):
            ale.plot(X_discrete, features=[1, 2, 3])
        with pytest.raises(
            TypeError, match="'features' must be an int or a list of int"
        ):
            ale.plot(X_discrete, features="invalid_feature_format")
        with pytest.raises(
            ValueError, match="Categorical not yet implemented"
        ):
            ale.plot(X_discrete, features=0, feature_type="categorical")
