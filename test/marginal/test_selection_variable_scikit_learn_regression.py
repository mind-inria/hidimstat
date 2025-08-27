from copy import deepcopy
import numpy as np
import pytest

from hidimstat import UnivariateLinearRegressionTests, MutualInformation


def configure_marginal_regression(ClassMethod, X, y):
    """
    Configure ClassMethod model for feature importance analysis.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix where each column represents a feature
        and each row a sample.
    y : array-like of shape (n_samples,)
        Target variable array.
    Returns
    -------
    importance : array-like
        Array containing importance scores for each feature.
        Higher values indicate greater feature importance in predicting
        the target variable.
    Notes
    -----
    The function performs the following steps:
    1. Intanciate ClassMethod
    2. Calculates feature importance
    """
    # instantiate model
    vi = ClassMethod()
    # fit the model using the training set
    vi.fit()
    # calculate feature importance using the test set
    importance = vi.importance(X, y)
    return np.array(importance)


parameter_exact = [
    ("HiDim", 150, 200, 1, 0.0, 42, 1.0, np.inf, 0.0),
    ("HiDim with noise", 150, 200, 1, 0.0, 42, 1.0, 10.0, 0.0),
    ("HiDim with correlated noise", 150, 200, 1, 0.0, 42, 1.0, 10.0, 0.5),
    ("HiDim with correlated features", 150, 200, 1, 0.8, 42, 1.0, np.inf, 0.0),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_exact))[1:])),
    ids=list(zip(*parameter_exact))[0],
)
@pytest.mark.parametrize(
    "ClassVI",
    [UnivariateLinearRegressionTests, MutualInformation],
    ids=["UnivariateLinearRegressionTests", "MutualInformation"],
)
def test_linear_data_exact(data_generator, ClassVI):
    """Tests the method on linear cases with noise and/or correlation"""
    X, y, important_features, _ = data_generator

    importance = configure_marginal_regression(ClassVI, X, y)
    # check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)
    # check that important features have the highest importance scores
    assert np.all([int(i) in important_features for i in np.argsort(importance)[-1:]])


parameter_bad_detection = [
    ("HiDim with high correlated features", 150, 200, 1, 1.0, 42, 1.0, 5.0, 0.0),
    ("HiDim multivaribale", 150, 200, 10, 0.0, 42, 1.0, np.inf, 0.0),
    ("HiDim multivaribale noise", 150, 200, 10, 0.0, 42, 1.0, 10.0, 0.0),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_bad_detection))[1:])),
    ids=list(zip(*parameter_bad_detection))[0],
)
@pytest.mark.parametrize(
    "ClassVI",
    [UnivariateLinearRegressionTests, MutualInformation],
    ids=["UnivariateLinearRegressionTests", "MutualInformation"],
)
def test_linear_data_fail(data_generator, ClassVI):
    """Tests the faillure of the method on linear cases with correlation
    or multiple variable of importance"""
    X, y, important_features, _ = data_generator
    size_support = np.sum(important_features != 0)

    importance = configure_marginal_regression(ClassVI, X, y)
    # check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)
    # check that important features have the highest importance scores
    assert np.any(
        [
            int(i) not in important_features
            for i in np.argsort(importance)[-size_support:]
        ]
    )


##############################################################################
# Spefic test for UnivariateLinearRegressionTests
parameter_exact_UnivariateLinearRegressionTests = [
    ("HiDim with high level noise", 150, 200, 1, 0.2, 42, 1.0, 0.5, 0.0),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_exact_UnivariateLinearRegressionTests))[1:])),
    ids=list(zip(*parameter_exact_UnivariateLinearRegressionTests))[0],
)
@pytest.mark.parametrize(
    "center, force_finite",
    [
        (True, True),
        (False, True),
        (True, False),
    ],
    ids=["default", "no center", "no force finite"],
)
def test_UnivariateLinearRegressionTests_exact(data_generator, center, force_finite):
    """Tests parameters of classes"""
    X, y, important_features, not_important_features = data_generator

    importance = (
        UnivariateLinearRegressionTests(center=center, force_finite=force_finite)
        .fit()
        .importance(X, y)
    )
    # check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)
    # check that important features have the highest importance scores
    assert np.all([int(i) in important_features for i in np.argsort(importance)[-1:]])
    # Check that important features have higher mean importance scores
    assert (
        importance[important_features].mean()
        > importance[not_important_features][
            np.where(importance[not_important_features] != 0)
        ].mean()
    )


# Spefic test for MutualInformation
@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [parameter_exact[0][1:]],
    ids=[parameter_exact[0][0]],
)
@pytest.mark.parametrize(
    "discrete_features, n_neighbors",
    [
        ("auto", 5),
        (False, 3),
    ],
    ids=["change number of neighboor", "discrete_features False"],
)
def test_MutualInformation_exact(data_generator, discrete_features, n_neighbors):
    """Tests parameters of classes"""
    X, y, important_features, _ = data_generator
    size_support = np.sum(important_features != 0)

    importance = (
        MutualInformation(discrete_features=discrete_features, n_neighbors=n_neighbors)
        .fit()
        .importance(X, y)
    )
    # check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)
    # check that important features have the highest importance scores
    assert np.all(
        [int(i) in important_features for i in np.argsort(importance)[-size_support:]]
    )


##############################################################################
@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(150, 200, 1, 0.0, 42, 1.0, 0.0, 0.0)],
    ids=["default data"],
)
@pytest.mark.parametrize(
    "ClassVI",
    [UnivariateLinearRegressionTests, MutualInformation],
    ids=["UnivariateLinearRegressionTests", "MutualInformation"],
)
class TestClass:
    """Test the element of the class"""

    def test_init(self, data_generator, ClassVI):
        """Test initialization work"""
        classvi = ClassVI()

    def test_fit(self, data_generator, ClassVI):
        """Test fitting is doing nothing"""
        classvi = ClassVI()
        classvi_reference = deepcopy(classvi)
        classvi.fit()
        for attribute_name in classvi.__dict__.keys():
            assert classvi.__getattribute__(
                attribute_name
            ) == classvi_reference.__getattribute__(attribute_name)

    def test_categorical(
        self,
        n_samples,
        n_features,
        support_size,
        rho,
        seed,
        value,
        signal_noise_ratio,
        rho_serial,
        ClassVI,
    ):
        """Test the fit_importance function on mix type of feature"""
        rng = np.random.default_rng(seed)
        X_cont = rng.random((n_samples, 2))
        X_cat = rng.integers(low=0, high=3, size=(n_samples, 1))
        X = np.hstack([X_cont, X_cat])
        y = rng.random((n_samples, 1))

        classvi = ClassVI()

        importances = classvi.fit_importance(X, y)
        assert len(importances) == 3
        assert np.all(importances >= 0)
