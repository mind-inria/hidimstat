from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

from hidimstat import CPI, BasePerturbation


def configure_linear_categorial_cpi(X, y, n_permutation, seed):
    """
    Configure Conditional Predictive Impact (CPI) model with linear regression
    for feature importance analysis.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix where each column represents a feature
        and each row a sample.
    y : array-like of shape (n_samples,)
        Target variable array.
    n_permutation : int
        Number of permutations to perform for the CPI analysis.
    seed : int
        Random seed for reproducibility.
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
    3. Configures CPI with linear regression as both estimator and imputer
    4. Calculates feature importance using the test set
    The CPI method uses permutation-based importance scoring with linear
    regression as the base model.
    """
    # split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # create and fit a linear regression model on the training set
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    # instantiate CPI model with linear regression imputer
    cpi = CPI(
        estimator=regression_model,
        imputation_model_continuous=LinearRegression(),
        n_permutations=n_permutation,
        method="predict",
        random_state=seed,
        n_jobs=1,
    )
    # fit the model using the training set
    cpi.fit(
        X_train,
        groups=None,
        var_type="auto",
    )
    # calculate feature importance using the test set
    vim = cpi.importance(X_test, y_test)
    importance = vim["importance"]
    return importance


##############################################################################
## tests cpi on different type of data
parameter_exact = [
    ("HiDim", 150, 200, 10, 0.0, 42, 1.0, np.inf, 0.0),
    ("HiDim with noise", 150, 200, 10, 0.0, 42, 1.0, 10.0, 0.0),
    ("HiDim with correlated noise", 150, 200, 10, 0.0, 42, 1.0, 10.0, 0.2),
    ("HiDim with correlated features", 150, 200, 10, 0.2, 42, 1.0, np.inf, 0.0),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, snr, rho_noise_time",
    zip(*(list(zip(*parameter_exact))[1:])),
    ids=list(zip(*parameter_exact))[0],
)
@pytest.mark.parametrize("cpi_n_permutation, cpi_seed", [(10, 0)], ids=["default_cpi"])
def test_cpi_linear_data_exact(data_generator, cpi_n_permutation, cpi_seed):
    """Tests the method on linear cases with noise and correlation"""
    X, y, important_features, _ = data_generator
    importance = configure_linear_categorial_cpi(X, y, cpi_n_permutation, cpi_seed)
    # check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)
    # check that important features have the highest importance scores
    assert np.all([int(i) in important_features for i in np.argsort(importance)[-10:]])


parameter_partial = [
    ("HiDim with correlated features and noise", 150, 200, 10, 0.2, 42, 1, 10, 0),
    (
        "HiDim with correlated features and correlated noise",
        150,
        200,
        10,
        0.2,
        42,
        1.0,
        10,
        0.2,
    ),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, snr, rho_noise_time",
    zip(*(list(zip(*parameter_partial))[1:])),
    ids=list(zip(*parameter_partial))[0],
)
@pytest.mark.parametrize("cpi_n_permutation, cpi_seed", [(10, 0)], ids=["default_cpi"])
def test_cpi_linear_data_partial(data_generator, cpi_n_permutation, cpi_seed):
    """Tests the method on linear cases with noise and correlation"""
    X, y, important_features, _ = data_generator
    importance = configure_linear_categorial_cpi(X, y, cpi_n_permutation, cpi_seed)
    # check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)
    # check that important features have the highest importance scores
    min_rank = 0
    importance_sort = np.flip(np.argsort(importance))
    for index in important_features:
        rank = np.where(importance_sort == index)[0]
        if rank > min_rank:
            min_rank = rank
    # accept missing ranking of 5 elements
    assert min_rank < 15


parameter_bad_detection = [
    ("high level noise", 150, 200, 10, 0.2, 42, 1.0, 1.0, 0.0),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, snr, rho_noise_time",
    zip(*(list(zip(*parameter_bad_detection))[1:])),
    ids=list(zip(*parameter_bad_detection))[0],
)
@pytest.mark.parametrize("cpi_n_permutation, cpi_seed", [(20, 0)], ids=["default_cpi"])
def test_cpi_linear_fail(data_generator, cpi_n_permutation, cpi_seed):
    """Tests when the method doesn't identify all important features"""
    X, y, important_features, not_important_features = data_generator
    importance = configure_linear_categorial_cpi(X, y, cpi_n_permutation, cpi_seed)
    # check that importance is defined for each feature
    assert importance.shape == (X.shape[1],)
    # check that mean importance of important features is
    # higher than mean importance of other features
    assert (
        importance[important_features].mean()
        > importance[not_important_features].mean()
    )
    # Verify that not all important features are detected
    assert np.sum(
        [int(i) in important_features for i in np.argsort(importance)[-10:]]
    ) != len(important_features)


##############################################################################
## Test specific options of cpi
@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, snr, rho_noise_time",
    [(150, 200, 10, 0.0, 42, 1.0, np.inf, 0.0)],
    ids=["high dimension"],
)
def test_cpi_group(data_generator):
    """Test CPI with groups using pandas objects"""
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

    cpi = CPI(
        estimator=regression_model,
        imputation_model_continuous=LinearRegression(),
        n_permutations=20,
        method="predict",
        random_state=0,
        n_jobs=1,
    )
    cpi.fit(
        X_train_df,
        groups=groups,
        var_type="continuous",
    )
    # Warning expected since column names in pandas are not considered
    with pytest.warns(UserWarning, match="X does not have valid feature names, but"):
        vim = cpi.importance(X_test_df, y_test)

    importance = vim["importance"]
    # Check if importance scores are computed for each feature
    assert importance.shape == (2,)
    # Verify that important feature group has higher score
    # than non-important feature group
    assert importance[0] > importance[1]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, snr, rho_noise_time",
    zip(*(list(zip(*parameter_exact))[1:])),
    ids=list(zip(*parameter_exact))[0],
)
def test_cpi_classication(data_generator):
    """Test CPI for a classification problem"""
    X, y, important_features, not_important_features = data_generator
    # Create categories
    y_clf = deepcopy(y)
    y_clf[np.where(y > 4)] = 0
    y_clf[np.where(np.logical_and(y <= 4, y > 0))] = 1
    y_clf[np.where(np.logical_and(y <= 0, y > -4))] = 2
    y_clf[np.where(y <= -4)] = 3
    y_clf = np.array(y_clf, dtype=int)

    # Split the data into training and test sets
    X_train, X_test, y_train_clf, y_test_clf = train_test_split(
        X, y_clf, random_state=0
    )

    # Create and fit a logistic regression model on the training set
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train_clf)

    cpi = CPI(
        estimator=logistic_model,
        imputation_model_continuous=LinearRegression(),
        n_permutations=20,
        random_state=0,
        n_jobs=1,
        method="predict_proba",
        loss=log_loss,
    )
    cpi.fit(
        X_train,
        groups=None,
        var_type=["continuous"] * X.shape[1],
    )
    vim = cpi.importance(X_test, y_test_clf)
    importance = vim["importance"]
    # Check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)
    # Check that important features have higher mean importance scores
    assert (
        importance[important_features].mean()
        > importance[not_important_features].mean()
    )


##############################################################################
@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, snr, rho_noise_time",
    [(150, 200, 10, 0.0, 42, 1.0, 0.0, 0.0)],
    ids=["default data"],
)
class TestCPIClass:
    """Test the element of the class"""

    def test_cpi_init(self, data_generator):
        """Test CPI initialization"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        cpi = CPI(
            estimator=fitted_model,
            method="predict",
        )
        assert cpi.n_jobs == 1
        assert cpi.n_permutations == 50
        assert cpi.loss == root_mean_squared_error
        assert cpi.method == "predict"
        assert cpi.categorical_max_cardinality == 10
        assert cpi.imputation_model_categorical is None
        assert cpi.imputation_model_continuous is None

    def test_cpi_fit(self, data_generator):
        """Test fitting CPI"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        cpi = CPI(
            estimator=fitted_model,
            imputation_model_continuous=LinearRegression(),
            random_state=42,
        )

        # Test fit with auto var_type
        cpi.fit(X)
        assert len(cpi._list_imputation_models) == X.shape[1]
        assert cpi.n_groups == X.shape[1]

        # Test fit with specified groups
        groups = {"g1": [0, 1], "g2": [2, 3, 4]}
        cpi.fit(X, groups=groups)
        assert len(cpi._list_imputation_models) == 2
        assert cpi.n_groups == 2

    def test_cpi_categorical(
        self, n_samples, n_features, support_size, rho, seed, value, snr, rho_noise_time
    ):
        """Test CPI with categorical variables"""
        rng = np.random.default_rng(seed)
        X_cont = rng.random((n_samples, 2))
        X_cat = rng.integers(low=0, high=3, size=(n_samples, 1))
        X = np.hstack([X_cont, X_cat])
        y = rng.random((n_samples, 1))
        fitted_model = LinearRegression().fit(X, y)

        cpi = CPI(
            estimator=fitted_model,
            imputation_model_continuous=LinearRegression(),
            imputation_model_categorical=LogisticRegression(),
            random_state=seed + 1,
        )

        var_type = ["continuous", "continuous", "categorical"]
        cpi.fit(X, y, var_type=var_type)

        importances = cpi.importance(X, y)["importance"]
        assert len(importances) == 3
        assert np.all(importances >= 0)


##############################################################################
@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, snr, rho_noise_time",
    [(150, 200, 10, 0.0, 42, 1.0, 0.0, 0.0)],
    ids=["default data"],
)
class TestCPIExceptions:
    """Test class for CPI exceptions"""

    def test_unfitted_estimator(self, data_generator):
        """Test when using an unfitted estimator"""
        with pytest.raises(NotFittedError):
            CPI(
                estimator=LinearRegression(),
                method="predict",
            )

    def test_unknown_predict_method(self, data_generator):
        """Test when an unknown prediction method is provided"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)

        with pytest.raises(ValueError):
            CPI(
                estimator=fitted_model,
                method="unknown method",
            )

    def test_unfitted_predict(self, data_generator):
        """Test predict method with unfitted model"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        cpi = CPI(
            estimator=fitted_model,
            method="predict",
        )

        with pytest.raises(ValueError, match="The class is not fitted."):
            cpi.predict(X)

    def test_unfitted_importance(self, data_generator):
        """Test importance method with unfitted model"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        cpi = CPI(
            estimator=fitted_model,
            method="predict",
        )

        with pytest.raises(ValueError, match="The class is not fitted."):
            cpi.importance(X, y)

    def test_unfitted_base_perturbation(self, data_generator):
        """Test base perturbation with unfitted estimators"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        cpi = CPI(
            estimator=fitted_model,
            method="predict",
        )
        BasePerturbation.fit(cpi, X, y)

        with pytest.raises(
            ValueError,
            match="The imputation models require to be fitted before being used.",
        ):
            cpi.importance(X, y)

    def test_cpi_errors(self, data_generator):
        """Test error handling"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        cpi = CPI(estimator=fitted_model)

        # Test error when passing invalid var_type
        with pytest.raises(ValueError, match="type of data 'invalid' unknow."):
            cpi.fit(X, var_type="invalid")

    def test_invalid_n_permutations(self, data_generator):
        """Test when invalid number of permutations is provided"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)

        with pytest.raises(AssertionError, match="n_permutations must be positive"):
            CPI(estimator=fitted_model, n_permutations=-1, method="predict")

    def test_invalid_n_jobs(self, data_generator):
        """Test when invalid number of jobs is provided"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)

        with pytest.raises(AssertionError, match="n_jobs must be positive"):
            CPI(estimator=fitted_model, n_jobs=0, method="predict")

    def test_mismatched_features(self, data_generator):
        """Test when number of features doesn't match between fit and predict"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        cpi = CPI(
            estimator=fitted_model,
            imputation_model_continuous=LinearRegression(),
            method="predict",
        )
        cpi.fit(X, groups=None, var_type="auto")

        with pytest.warns(
            UserWarning, match="Not all features will has a importance score."
        ):
            cpi.importance(X[:, :-1], y)

    def test_invalid_var_type(self, data_generator):
        """Test when invalid variable type is provided"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        cpi = CPI(estimator=fitted_model, method="predict")

        with pytest.raises(ValueError, match="type of data 'invalid_type' unknow."):
            cpi.fit(X, groups=None, var_type=["invalid_type"] * X.shape[1])

    def test_incompatible_imputer(self, data_generator):
        """Test when incompatible imputer is provided"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)

        with pytest.raises(AssertionError, match="Continous imputation model invalid"):
            cpi = CPI(
                estimator=fitted_model,
                imputation_model_continuous="invalid_imputer",
                method="predict",
            )
            cpi.fit(X, y)

        with pytest.raises(AssertionError, match="Categorial imputation model invalid"):
            cpi = CPI(
                estimator=fitted_model,
                imputation_model_categorical="invalid_imputer",
                method="predict",
            )
            cpi.fit(X, y)

    def test_invalid_groups_format(self, data_generator):
        """Test when groups are provided in invalid format"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        cpi = CPI(estimator=fitted_model, method="predict")

        invalid_groups = ["group1", "group2"]  # Should be dictionary
        with pytest.raises(ValueError, match="groups needs to be a dictionnary"):
            cpi.fit(X, groups=invalid_groups, var_type="auto")

    def test_groups_warning(self, data_generator):
        """Test if a subgroup raise a warning"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        cpi = CPI(
            estimator=fitted_model,
            imputation_model_continuous=LinearRegression(),
            method="predict",
        )
        subgroups = {"group1": [0, 1], "group2": [2, 3]}
        cpi.fit(X, y, groups=subgroups, var_type="auto")

        with pytest.warns(
            UserWarning, match="Not all features will has a importance score."
        ):
            cpi.importance(X, y)
