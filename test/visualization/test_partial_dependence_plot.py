import matplotlib
import matplotlib.pyplot as plt
import pytest
from sklearn.linear_model import LinearRegression

from hidimstat._utils.scenario import multivariate_simulation
from hidimstat.visualization import PDP

matplotlib.use("Agg")


parameters_pdp = [("pdp", 200, 20, 3, 0.3, 92, 1.0, 10.0, 0.0)]

pdp_ids = [p[0] for p in parameters_pdp]
pdp_values = [p[1:] for p in parameters_pdp]


@pytest.fixture(scope="module")
def pdp_test_data():
    """Fixture to generate test data and a fitted LinearRegression model for PDP tests."""
    X, y, _, _ = multivariate_simulation(
        n_samples=200,
        n_features=20,
        support_size=3,
        rho=0.3,
        value=1.0,
        signal_noise_ratio=10.0,
        rho_serial=0.0,
        shuffle=False,
        seed=92,
    )

    model = LinearRegression()
    model.fit(X, y)

    return X, y, model


def test_pdp_smoke(pdp_test_data):
    """Test 1D and 2D PDP plotting functionality."""
    X, _, model = pdp_test_data

    feature_names = [f"feat_{i}" for i in range(X.shape[1])]
    pdp = PDP(model, feature_names=feature_names)

    axes_1d = pdp.plot(X, features=0)
    assert axes_1d is not None
    plt.close("all")

    axes_2d = pdp.plot(X, features=[0, 1], cmap="plasma")
    assert axes_2d is not None
    plt.close("all")


def test_pdp_no_feature_names(pdp_test_data):
    """Test PDP plotting default feature names generation."""
    X, _, model = pdp_test_data

    pdp = PDP(model)

    axes_1d = pdp.plot(X, features=0)
    assert axes_1d is not None
    plt.close("all")


def test_pdp_invalid_features_length(pdp_test_data):
    """Check the raised error of the PDP."""
    X, _, model = pdp_test_data

    pdp = PDP(model)

    with pytest.raises(
        ValueError, match="Only 1D and 2D PDP plots are supported"
    ):
        pdp.plot(X, features=[0, 1, 2])
