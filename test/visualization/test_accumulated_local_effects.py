import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from hidimstat.visualization import ALE
from hidimstat.visualization.accumulated_local_effects import (
    compute_ale_1d_continuous,
    compute_ale_1d_discrete,
    compute_ale_2d,
)

matplotlib.use("Agg")

parameters_ale = [("ale", 300, 20, 3, 0.3, 92, 1.0, 10.0, 0.0)]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameters_ale, strict=False))[1:]), strict=False),
    ids=next(zip(*parameters_ale, strict=False)),
)
class TestALE:
    def test_compute_ale_1d_continuous(self, data_generator):
        X, y, important_features, _ = data_generator

        model = LinearRegression()
        model.fit(X, y)

        grid_resolution = 10

        result = compute_ale_1d_continuous(
            model,
            X,
            feature_idx=important_features[0],
            grid_resolution=grid_resolution,
        )

        assert "ale" in result
        assert "quantiles" in result
        assert isinstance(result["ale"], np.ndarray)
        assert len(result["ale"]) == len(result["quantiles"])
        assert len(result["quantiles"]) <= grid_resolution + 1

    def test_compute_ale_1d_discrete(self, data_generator):
        X, y, _, _ = data_generator

        X_discrete = X.copy()
        X_discrete[:, 0] = np.random.default_rng(92).choice(
            [0, 1, 2], size=X.shape[0]
        )

        model = LinearRegression()
        model.fit(X, y)

        result = compute_ale_1d_discrete(model, X_discrete, feature_idx=0)

        assert "ale" in result
        assert "unique_values" in result
        assert len(result["ale"]) == len(result["unique_values"])
        assert len(result["unique_values"]) <= 3

    def test_compute_ale_2d(self, data_generator):
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
        assert result["ale"].shape == (
            len(result["quantiles_i"]),
            len(result["quantiles_j"]),
        )

    def test_ale_auto_feature_type_resolution(self, data_generator):
        X, y, _, _ = data_generator

        X_discrete = X.copy()
        X_discrete[:, 0] = np.random.default_rng(92).choice(
            [0, 1, 2], size=X.shape[0]
        )

        model = LinearRegression()
        model.fit(X_discrete, y)

        ale = ALE(model)

        assert (
            ale._resolve_feature_type(
                X_discrete, feature=0, feature_type="auto"
            )
            == "discrete"
        )
        assert (
            ale._resolve_feature_type(X, feature=0, feature_type="auto")
            == "continuous"
        )

    def test_ale_plot_smoke(self, data_generator):
        X, y, _, _ = data_generator

        X_discrete = X.copy()
        X_discrete[:, 0] = np.random.default_rng(92).choice(
            [0, 1, 2], size=X.shape[0]
        )

        model = LinearRegression()
        model.fit(X_discrete, y)

        ale = ALE(model)

        ax_1d_continuous = ale.plot(X_discrete, features=1, grid_resolution=10)
        assert ax_1d_continuous is not None
        plt.close("all")

        ax_1d_discrete = ale.plot(X_discrete, features=0)
        assert ax_1d_discrete is not None
        plt.close("all")

        ax_2d = ale.plot(X_discrete, features=[1, 2], grid_resolution=5)
        assert ax_2d is not None
        plt.close("all")
