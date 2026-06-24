import matplotlib
import matplotlib.pyplot as plt
import pytest
from sklearn.linear_model import LinearRegression

from hidimstat.visualization import PDP

matplotlib.use("Agg")


parameters_pdp = [("pdp", 200, 20, 3, 0.3, 92, 1.0, 10.0, 0.0)]

ale_ids = [p[0] for p in parameters_pdp]
ale_values = [p[1:] for p in parameters_pdp]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    ale_values,
    ids=ale_ids,
)
class TestPDP:
    def test_pdp_smoke(self, data_generator):
        """Test 1D and 2D PDP plotting functionality."""
        X, y, _, _ = data_generator

        model = LinearRegression()
        model.fit(X, y)

        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        pdp = PDP(model, feature_names=feature_names)

        axes_1d = pdp.plot(X, features=0)
        assert axes_1d is not None
        plt.close("all")

        axes_2d = pdp.plot(X, features=[0, 1], cmap="plasma")
        assert axes_2d is not None
        plt.close("all")

    def test_pdp_no_feature_names(self, data_generator):
        """Test PDP plotting default feature names generation."""
        X, y, _, _ = data_generator

        model = LinearRegression()
        model.fit(X, y)

        pdp = PDP(model)

        axes_1d = pdp.plot(X, features=0)
        assert axes_1d is not None
        plt.close("all")

    def test_pdp_invalid_features_length(self, data_generator):
        """Check the raised error of the PDP."""
        X, y, _, _ = data_generator

        model = LinearRegression()
        model.fit(X, y)

        pdp = PDP(model)

        with pytest.raises(
            ValueError, match="Only 1D and 2D PDP plots are supported"
        ):
            pdp.plot(X, features=[0, 1, 2])
