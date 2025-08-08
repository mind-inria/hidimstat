from copy import deepcopy
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression

from hidimstat import PDP


def configure_linear_categorial_pdp(X, y):
    """
    Configure Partial Dependance Plot (PDP) model with linear regression
    for feature importance analysis.
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
    1. Fits a linear regression model on training data
    2. Configures PDP with linear regression
    3. Calculates feature importance using the test set
    The PDP method is a marginal methods scoring with linear
    regression as the base model.
    """
    # create and fit a linear regression model on the training set
    regression_model = LinearRegression()
    regression_model.fit(X, y)

    # instantiate CPI model with linear regression imputer
    pdp = PDP(
        estimator=regression_model,
        n_jobs=1,
    )
    # calculate feature importance using the test set
    importance = pdp.importance(X)
    return np.array(importance)


parameter_exact = [
    ("HiDim", 150, 200, 1, 0.0, 42, 1.0, np.inf, 0.0),
    ("HiDim with noise", 150, 200, 1, 0.0, 42, 1.0, 10.0, 0.0),
    ("HiDim with correlated noise", 150, 200, 1, 0.0, 42, 1.0, 10.0, 0.5),
    ("HiDim with correlated features", 150, 200, 1, 0.8, 42, 1.0, np.inf, 0.0),
    ("HiDim with high level noise", 150, 200, 10, 0.2, 42, 1.0, 0.5, 0.0),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_exact))[1:])),
    ids=list(zip(*parameter_exact))[0],
)
def test_pdp_linear_data_exact(data_generator):
    """Tests the method on linear cases with noise and correlation"""
    X, y, important_features, _ = data_generator

    importance = configure_linear_categorial_pdp(X, y)
    # check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)
    # check that important features have the highest importance scores
    assert np.all([int(i) in important_features for i in np.argsort(importance)[-1:]])


parameter_bad_detection = [
    ("HiDim with high correlated features", 150, 200, 1, 1.0, 42, 1.0, 5.0, 0.0),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_bad_detection))[1:])),
    ids=list(zip(*parameter_bad_detection))[0],
)
def test_pdp_linear_data_fail(data_generator):
    """Tests the method on linear cases with noise and correlation"""
    X, y, important_features, _ = data_generator

    importance = configure_linear_categorial_pdp(X, y)
    # check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)
    # check that important features have the highest importance scores
    assert np.any(
        [int(i) not in important_features for i in np.argsort(importance)[-1:]]
    )


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_exact))[1:])),
    ids=list(zip(*parameter_exact))[0],
)
def test_pdp_classication(data_generator):
    """Test PDP for a classification problem"""
    X, y, important_features, not_important_features = data_generator
    # Create categories
    y_clf = deepcopy(y)
    y_clf[np.where(y > 2)] = 0
    y_clf[np.where(np.logical_and(y <= 2, y > 0))] = 1
    y_clf[np.where(np.logical_and(y <= 0, y > -2))] = 2
    y_clf[np.where(y <= -2)] = 3
    y_clf = np.array(y_clf, dtype=int)

    # Create and fit a logistic regression model on the training set
    logistic_model = LogisticRegression()
    logistic_model.fit(X, y_clf)

    pdp = PDP(
        estimator=logistic_model,
        n_jobs=1,
        method="predict_proba",
    )
    importance = pdp.fit_importance(X)

    # Check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)
    # Check that important features have higher mean importance scores
    assert (
        importance[important_features].mean()
        > importance[not_important_features].mean()
    )


##############################################################################
@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(150, 200, 1, 0.0, 42, 1.0, 0.0, 0.0)],
    ids=["default data"],
)
class TestPDPClass:
    """Test the element of the class"""

    def test_pdp_init(self, data_generator):
        """Test PDP initialization"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(
            estimator=fitted_model,
            method="predict",
        )
        assert pdp.n_jobs == 1
        assert pdp.method == "predict"

    def test_pdp_importance(self, data_generator):
        """Test fitting PDP"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(
            estimator=fitted_model,
        )
        # Test importance
        pdp.importance(X)
        assert len(pdp.importances_) > 0
        assert len(pdp.ices_) > 0
        assert pdp.pvalues_ is None

    def test_pdp_categorical(
        self,
        n_samples,
        n_features,
        support_size,
        rho,
        seed,
        value,
        signal_noise_ratio,
        rho_serial,
    ):
        """Test PDP with categorical variables"""
        rng = np.random.default_rng(seed)
        X_cont = rng.random((n_samples, 2))
        X_cat = np.concatenate(
            [
                rng.integers(low=0, high=3, size=(n_samples, 1)),
                rng.integers(low=0, high=5, size=(n_samples, 1)),
            ],
            axis=1,
        )
        X = np.hstack([X_cont, X_cat])
        categories_features = np.ones(X.shape[1], dtype=bool)
        categories_features[: X_cont.shape[1]] = False
        y = rng.random((n_samples, 1))
        fitted_model = LinearRegression().fit(X, y)

        pdp = PDP(estimator=fitted_model, categorical_features=categories_features)

        importances = pdp.fit_importance(X)
        assert len(importances) == 4
        assert np.all(importances < 0.1)  # no informative, worse than dummy model


##############################################################################
@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(150, 200, 10, 0.0, 42, 1.0, 0.0, 0.0)],
    ids=["default data"],
)
class TestPDPExceptions:
    """Test class for PDP exceptions"""

    def test_unfitted_estimator(self, data_generator):
        """Test when using an unfitted estimator"""
        with pytest.raises(NotFittedError):
            PDP(
                estimator=LinearRegression(),
                method="predict",
            )

    def test_unknown_predict_method(self, data_generator):
        """Test when an unknown prediction method is provided"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)

        with pytest.raises(ValueError):
            PDP(
                estimator=fitted_model,
                method="unknown method",
            )

    def test_unfitted_importance(self, data_generator):
        """Test importance method with unfitted model"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(
            estimator=fitted_model,
        )

        with pytest.raises(
            ValueError, match="The importances need to be called before."
        ):
            pdp.plot(feature_id=0, X=X)
