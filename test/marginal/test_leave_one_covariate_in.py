from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

from hidimstat import LOCI


def configure_linear_categorial_loci(X, y):
    """
    Configure Leave One Covariate In (LOCI) model with linear regression
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
    1. Splits data into training and test sets
    2. Fits a linear regression model on training data
    3. Configures LOCI with linear regression
    4. Calculates feature importance using the test set
    The LOCI method is a marginal methods scoring with linear
    regression as the base model.
    """
    # split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # create and fit a linear regression model on the training set
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    # instantiate LOCI model with linear regression imputer
    loci = LOCI(
        estimator=regression_model,
        method="predict",
        n_jobs=1,
    )
    # fit the model using the training set
    loci.fit(
        X_train,
        y_train,
        groups=None,
    )
    # calculate feature importance using the test set
    importance = loci.importance(X_test, y_test)
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
def test_loci_linear_data_exact(data_generator):
    """Tests the method on linear cases with noise and correlation"""
    X, y, important_features, _ = data_generator

    importance = configure_linear_categorial_loci(X, y)
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
def test_loci_linear_data_fail(data_generator):
    """Tests the method on linear cases with noise and correlation"""
    X, y, important_features, _ = data_generator

    importance = configure_linear_categorial_loci(X, y)
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
def test_loci_classication(data_generator):
    """Test LOCI for a classification problem"""
    X, y, important_features, not_important_features = data_generator
    # Create categories
    y_clf = deepcopy(y)
    y_clf[np.where(y > 4)] = 0
    y_clf[np.where(np.logical_and(y <= 4, y > 0))] = 1
    y_clf[np.where(np.logical_and(y <= 0, y > -4))] = 2
    y_clf[np.where(y <= -4)] = 3
    y_clf = np.array(y_clf, dtype=int)

    # Create and fit a logistic regression model on the training set
    logistic_model = LogisticRegression()
    logistic_model.fit(X, y_clf)

    loci = LOCI(
        estimator=logistic_model,
        n_jobs=1,
        method="predict_proba",
        loss=log_loss,
    )
    importance = loci.fit_importance(
        X,
        y_clf,
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
    )

    # Check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)
    # Check that important features have higher mean importance scores
    assert (
        importance[important_features].mean()
        > importance[not_important_features].mean()
    )


##############################################################################
## Test specific options of loci
@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(150, 200, 10, 0.0, 42, 1.0, np.inf, 0.0)],
    ids=["high dimension"],
)
def test_loci_group(data_generator):
    """Test LOCI with groups using pandas objects"""
    X, y, important_features, not_important_features = data_generator

    # Create groups and convert to pandas DataFrame
    groups = {
        "group_0": [f"col_{i}" for i in important_features],
        "the_group_1": [f"col_{i}" for i in not_important_features],
    }
    X_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
    # Split data into training and test sets
    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, random_state=0)

    # Create and fit linear regression model on training set
    regression_model = LinearRegression()
    regression_model.fit(X_train_df, y_train)

    loci = LOCI(
        estimator=regression_model,
        method="predict",
        n_jobs=1,
    )
    loci.fit(
        X_train_df,
        y_train,
        groups=groups,
    )
    importance = np.array(loci.importance(X_test_df, y_test))

    # Check if importance scores are computed for each feature
    assert importance.shape == (2,)
    # Verify that important feature group has higher score
    # than non-important feature group
    assert importance[0] > importance[1]


##############################################################################
@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(150, 200, 1, 0.0, 42, 1.0, 0.0, 0.0)],
    ids=["default data"],
)
class TestLOCIClass:
    """Test the element of the class"""

    def test_loci_init(self, data_generator):
        """Test LOCI initialization"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        loci = LOCI(
            estimator=fitted_model,
            method="predict",
        )
        assert loci.n_jobs == 1
        assert loci.loss == root_mean_squared_error
        assert loci.method == "predict"

    def test_loci_fit(self, data_generator):
        """Test fitting LOCI"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        loci = LOCI(
            estimator=fitted_model,
        )

        # Test fit with auto var_type
        loci.fit(X, y)
        assert len(loci._list_univariate_model) == X.shape[1]
        assert loci.n_groups == X.shape[1]

        # Test fit with specified groups
        groups = {"g1": [0, 1], "g2": [2, 3, 4]}
        loci.fit(X, y, groups=groups)
        assert len(loci._list_univariate_model) == 2
        assert loci.n_groups == 2

    def test_loci_categorical(
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
        """Test LOCI with categorical variables"""
        rng = np.random.default_rng(seed)
        X_cont = rng.random((n_samples, 2))
        X_cat = rng.integers(low=0, high=3, size=(n_samples, 1))
        X = np.hstack([X_cont, X_cat])
        y = rng.random((n_samples, 1))
        fitted_model = LinearRegression().fit(X, y)

        loci = LOCI(
            estimator=fitted_model,
        )

        importances = loci.fit_importance(X, y, cv=KFold())
        assert len(importances) == 3
        assert np.all(importances < 0)  # no informative, worse than dummy model


##############################################################################
@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(150, 200, 10, 0.0, 42, 1.0, 0.0, 0.0)],
    ids=["default data"],
)
class TestLOCIExceptions:
    """Test class for LOCI exceptions"""

    def test_unfitted_estimator(self, data_generator):
        """Test when using an unfitted estimator"""
        with pytest.raises(NotFittedError):
            LOCI(
                estimator=LinearRegression(),
                method="predict",
            )

    def test_unknown_predict_method(self, data_generator):
        """Test when an unknown prediction method is provided"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)

        with pytest.raises(ValueError):
            LOCI(
                estimator=fitted_model,
                method="unknown method",
            )

    def test_unfitted_predict(self, data_generator):
        """Test predict method with unfitted model"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        loci = LOCI(
            estimator=fitted_model,
            method="predict",
        )

        with pytest.raises(ValueError, match="The class is not fitted."):
            loci.predict(X)

    def test_unfitted_importance(self, data_generator):
        """Test importance method with unfitted model"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        loci = LOCI(
            estimator=fitted_model,
            method="predict",
        )

        with pytest.raises(ValueError, match="The class is not fitted."):
            loci.importance(X, y)

    def test_invalid_groups_format(self, data_generator):
        """Test when groups are provided in invalid format"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        loci = LOCI(estimator=fitted_model, method="predict")

        invalid_groups = ["group1", "group2"]  # Should be dictionary
        with pytest.raises(ValueError, match="groups needs to be a dictionnary"):
            loci.fit(X, y, groups=invalid_groups)

    def test_groups_warning(self, data_generator):
        """Test if a subgroup raise a warning"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        loci = LOCI(
            estimator=fitted_model,
            method="predict",
        )
        subgroups = {"group1": [0, 1], "group2": [2, 3]}
        loci.fit(X, y, groups=subgroups)

        with pytest.warns(
            UserWarning,
            match="The importance will be computed only for features in the groups.",
        ):
            loci.importance(X, y)
