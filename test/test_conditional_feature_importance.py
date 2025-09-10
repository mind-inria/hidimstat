from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

from hidimstat import CFI
from hidimstat.base_perturbation import BasePerturbation
from hidimstat._utils.exception import InternalError
from hidimstat._utils.scenario import multivariate_simulation


def run_cfi(X, y, n_permutation, seed):
    """
    Configure Conditional Feature Importance (CFI) model with linear regression
    for feature importance analysis.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix where each column represents a feature
        and each row a sample.
    y : array-like of shape (n_samples,)
        Target variable array.
    n_permutation : int
        Number of permutations to perform for the CFI analysis.
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
    3. Configures CFI with linear regression as both estimator and imputer
    4. Calculates feature importance using the test set
    The CFI method uses permutation-based importance scoring with linear
    regression as the base model.
    """
    # split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # create and fit a linear regression model on the training set
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    # instantiate CFI model with linear regression imputer
    cfi = CFI(
        estimator=regression_model,
        imputation_model_continuous=LinearRegression(),
        n_permutations=n_permutation,
        method="predict",
        random_state=seed,
        n_jobs=1,
    )
    # fit the model using the training set
    cfi.fit(
        X_train,
        groups=None,
        var_type="auto",
    )
    # calculate feature importance using the test set
    vim = cfi.importance(X_test, y_test)
    importance = vim["importance"]
    return importance


##############################################################################
## tests cfi on different type of data
parameter_exact = [
    ("HiDim", 150, 200, 10, 0.0, 42, 1.0, np.inf, 0.0),
    ("HiDim with noise", 150, 200, 10, 0.0, 42, 1.0, 10.0, 0.0),
    ("HiDim with correlated noise", 150, 200, 10, 0.0, 42, 1.0, 10.0, 0.2),
    ("HiDim with correlated features", 150, 200, 10, 0.2, 42, 1.0, np.inf, 0.0),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_exact))[1:])),
    ids=list(zip(*parameter_exact))[0],
)
@pytest.mark.parametrize("n_permutation, cfi_seed", [(10, 0)], ids=["default_cfi"])
def test_linear_data_exact(data_generator, n_permutation, cfi_seed):
    """Tests the method on linear cases with noise and correlation"""
    X, y, important_features, _ = data_generator
    importance = run_cfi(X, y, n_permutation, cfi_seed)
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
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_partial))[1:])),
    ids=list(zip(*parameter_partial))[0],
)
@pytest.mark.parametrize("n_permutation, cfi_seed", [(10, 0)], ids=["default_cfi"])
def test_linear_data_partial(data_generator, n_permutation, cfi_seed):
    """Tests the method on linear cases with noise and correlation"""
    X, y, important_features, _ = data_generator
    importance = run_cfi(X, y, n_permutation, cfi_seed)
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


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(150, 200, 10, 0.2, 42, 1.0, 1.0, 0.0)],
    ids=["high level noise"],
)
@pytest.mark.parametrize("n_permutation, cfi_seed", [(20, 0)], ids=["default_cfi"])
def test_linear_data_fail(data_generator, n_permutation, cfi_seed):
    """Tests when the method doesn't identify all important features"""
    X, y, important_features, not_important_features = data_generator
    importance = run_cfi(X, y, n_permutation, cfi_seed)
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
## Test specific options of cfi
@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(150, 200, 10, 0.0, 42, 1.0, np.inf, 0.0)],
    ids=["high dimension"],
)
def test_group(data_generator):
    """Test CFI with groups using pandas objects"""
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

    cfi = CFI(
        estimator=regression_model,
        imputation_model_continuous=LinearRegression(),
        n_permutations=20,
        method="predict",
        random_state=0,
        n_jobs=1,
    )
    cfi.fit(
        X_train_df,
        groups=groups,
        var_type="continuous",
    )
    # Warning expected since column names in pandas are not considered
    with pytest.warns(UserWarning, match="X does not have valid feature names, but"):
        vim = cfi.importance(X_test_df, y_test)

    importance = vim["importance"]
    # Check if importance scores are computed for each feature
    assert importance.shape == (2,)
    # Verify that important feature group has higher score
    # than non-important feature group
    assert importance[0] > importance[1]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_exact))[1:])),
    ids=list(zip(*parameter_exact))[0],
)
def test_classication(data_generator):
    """Test CFI for a classification problem"""
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

    cfi = CFI(
        estimator=logistic_model,
        imputation_model_continuous=LinearRegression(),
        n_permutations=20,
        random_state=0,
        n_jobs=1,
        method="predict_proba",
        loss=log_loss,
    )
    cfi.fit(
        X_train,
        groups=None,
        var_type=["continuous"] * X.shape[1],
    )
    vim = cfi.importance(X_test, y_test_clf)
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
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(150, 200, 10, 0.0, 42, 1.0, 0.0, 0.0)],
    ids=["default data"],
)
class TestCFIClass:
    """Test the element of the class"""

    def test_init(self, data_generator):
        """Test CFI initialization"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        cfi = CFI(
            estimator=fitted_model,
            method="predict",
        )
        assert cfi.n_jobs == 1
        assert cfi.n_permutations == 50
        assert cfi.loss == root_mean_squared_error
        assert cfi.method == "predict"
        assert cfi.categorical_max_cardinality == 10
        assert cfi.imputation_model_categorical is None
        assert cfi.imputation_model_continuous is None

    def test_fit(self, data_generator):
        """Test fitting CFI"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        cfi = CFI(
            estimator=fitted_model,
            imputation_model_continuous=LinearRegression(),
            random_state=42,
        )

        # Test fit with auto var_type
        cfi.fit(X)
        assert len(cfi._list_imputation_models) == X.shape[1]
        assert cfi.n_groups == X.shape[1]

        # Test fit with specified groups
        groups = {"g1": [0, 1], "g2": [2, 3, 4]}
        cfi.fit(X, groups=groups)
        assert len(cfi._list_imputation_models) == 2
        assert cfi.n_groups == 2

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
    ):
        """Test CFI with categorical variables"""
        rng = np.random.default_rng(seed)
        X_cont = rng.random((n_samples, 2))
        X_cat = rng.integers(low=0, high=3, size=(n_samples, 1))
        X = np.hstack([X_cont, X_cat])
        y = rng.random((n_samples, 1))
        fitted_model = LinearRegression().fit(X, y)

        cfi = CFI(
            estimator=fitted_model,
            imputation_model_continuous=LinearRegression(),
            imputation_model_categorical=LogisticRegression(),
            random_state=seed + 1,
        )

        var_type = ["continuous", "continuous", "categorical"]
        cfi.fit(X, y, var_type=var_type)

        importances = cfi.importance(X, y)["importance"]
        assert len(importances) == 3
        assert np.all(importances >= 0)


##############################################################################
@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(150, 200, 10, 0.0, 42, 1.0, 0.0, 0.0)],
    ids=["default data"],
)
class TestCFIExceptions:
    """Test class for CFI exceptions"""

    def test_unfitted_estimator(self, data_generator):
        """Test when using an unfitted estimator"""
        with pytest.raises(NotFittedError):
            CFI(
                estimator=LinearRegression(),
                method="predict",
            )

    def test_unknown_predict_method(self, data_generator):
        """Test when an unknown prediction method is provided"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)

        with pytest.raises(ValueError):
            CFI(
                estimator=fitted_model,
                method="unknown method",
            )

    def test_unfitted_predict(self, data_generator):
        """Test predict method with unfitted model"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        cfi = CFI(
            estimator=fitted_model,
            method="predict",
        )

        with pytest.raises(ValueError, match="The class is not fitted."):
            cfi.predict(X)

    def test_unfitted_importance(self, data_generator):
        """Test importance method with unfitted model"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        cfi = CFI(
            estimator=fitted_model,
            method="predict",
        )

        with pytest.raises(ValueError, match="The class is not fitted."):
            cfi.importance(X, y)

    def test_unfitted_base_perturbation(self, data_generator):
        """Test base perturbation with unfitted estimators"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        cfi = CFI(
            estimator=fitted_model,
            method="predict",
        )
        BasePerturbation.fit(cfi, X, y)

        with pytest.raises(
            ValueError,
            match="The imputation models require to be fitted before being used.",
        ):
            cfi.importance(X, y)

    def test_invalid_type(self, data_generator):
        """Test invalid type of data"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        cfi = CFI(estimator=fitted_model)

        # Test error when passing invalid var_type
        with pytest.raises(ValueError, match="type of data 'invalid' unknow."):
            cfi.fit(X, var_type="invalid")

    def test_invalid_n_permutations(self, data_generator):
        """Test when invalid number of permutations is provided"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)

        with pytest.raises(AssertionError, match="n_permutations must be positive"):
            CFI(estimator=fitted_model, n_permutations=-1, method="predict")

    def test_not_good_type_X(self, data_generator):
        """Test when X is wrong type"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        cfi = CFI(
            estimator=fitted_model,
            imputation_model_continuous=LinearRegression(),
            method="predict",
        )
        cfi.fit(X, groups=None, var_type="auto")

        with pytest.raises(
            ValueError, match="X should be a pandas dataframe or a numpy array."
        ):
            cfi.importance(X.tolist(), y)

    def test_mismatched_features(self, data_generator):
        """Test when number of features doesn't match between fit and predict"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        cfi = CFI(
            estimator=fitted_model,
            imputation_model_continuous=LinearRegression(),
            method="predict",
        )
        cfi.fit(X, groups=None, var_type="auto")

        with pytest.raises(
            AssertionError, match="X does not correspond to the fitting data."
        ):
            cfi.importance(X[:, :-1], y)

    def test_mismatched_features_string(self, data_generator):
        """Test when name of features doesn't match between fit and predict"""
        X, y, _, _ = data_generator
        X = pd.DataFrame({"col_" + str(i): X[:, i] for i in range(X.shape[1])})
        fitted_model = LinearRegression().fit(X, y)
        cfi = CFI(
            estimator=fitted_model,
            imputation_model_continuous=LinearRegression(),
            method="predict",
        )
        subgroups = {
            "group1": ["col_" + str(i) for i in range(int(X.shape[1] / 2))],
            "group2": [
                "col_" + str(i) for i in range(int(X.shape[1] / 2), X.shape[1] - 3)
            ],
        }
        cfi.fit(X, groups=subgroups, var_type="auto")

        with pytest.raises(
            AssertionError,
            match=f"The array is missing at least one of the following columns \['col_100', 'col_101', 'col_102',",
        ):
            cfi.importance(
                X[np.concatenate([subgroups["group1"], subgroups["group2"][:-2]])], y
            )

    def test_internal_error(self, data_generator):
        """Test when name of features doesn't match between fit and predict"""
        X, y, _, _ = data_generator
        X = pd.DataFrame({"col_" + str(i): X[:, i] for i in range(X.shape[1])})
        fitted_model = LinearRegression().fit(X, y)
        cfi = CFI(
            estimator=fitted_model,
            imputation_model_continuous=LinearRegression(),
            method="predict",
        )
        subgroups = {
            "group1": ["col_" + str(i) for i in range(int(X.shape[1] / 2))],
            "group2": [
                "col_" + str(i) for i in range(int(X.shape[1] / 2), X.shape[1] - 3)
            ],
        }
        cfi.fit(X, groups=subgroups, var_type="auto")
        cfi.groups["group1"] = [None for i in range(100)]

        X = X.to_records(index=False)
        X = np.array(X, dtype=X.dtype.descr)
        with pytest.raises(
            InternalError,
            match=f"A problem with indexing has happened during the fit.",
        ):
            cfi.importance(X, y)

    def test_invalid_var_type(self, data_generator):
        """Test when invalid variable type is provided"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        cfi = CFI(estimator=fitted_model, method="predict")

        with pytest.raises(ValueError, match="type of data 'invalid_type' unknow."):
            cfi.fit(X, groups=None, var_type=["invalid_type"] * X.shape[1])

    def test_incompatible_imputer(self, data_generator):
        """Test when incompatible imputer is provided"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)

        with pytest.raises(AssertionError, match="Continous imputation model invalid"):
            cfi = CFI(
                estimator=fitted_model,
                imputation_model_continuous="invalid_imputer",
                method="predict",
            )
            cfi.fit(X, y)

        with pytest.raises(AssertionError, match="Categorial imputation model invalid"):
            cfi = CFI(
                estimator=fitted_model,
                imputation_model_categorical="invalid_imputer",
                method="predict",
            )
            cfi.fit(X, y)

    def test_invalid_groups_format(self, data_generator):
        """Test when groups are provided in invalid format"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        cfi = CFI(estimator=fitted_model, method="predict")

        invalid_groups = ["group1", "group2"]  # Should be dictionary
        with pytest.raises(ValueError, match="groups needs to be a dictionnary"):
            cfi.fit(X, groups=invalid_groups, var_type="auto")

    def test_groups_warning(self, data_generator):
        """Test if a subgroup raise a warning"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        cfi = CFI(
            estimator=fitted_model,
            imputation_model_continuous=LinearRegression(),
            method="predict",
        )
        subgroups = {"group1": [0, 1], "group2": [2, 3]}
        cfi.fit(X, y, groups=subgroups, var_type="auto")

        with pytest.warns(
            UserWarning,
            match="The number of features in X: 200 differs from the"
            " number of features for which importance is computed: 4",
        ):
            cfi.importance(X, y)


@pytest.fixture(scope="module")
def cfi_test_data():
    """
    Fixture to generate test data and a fitted LinearRegression model for CFI
    reproducibility tests.
    """
    X, y, _, _ = multivariate_simulation(
        n_samples=100,
        n_features=5,
        support_size=2,
        rho=0,
        value=1,
        signal_noise_ratio=4,
        rho_serial=0,
        shuffle=False,
        seed=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, model


def test_cfi_multiple_calls_are_repeatibility(cfi_test_data):
    """
    Test that multiple calls of .importance() when CFI is seeded provide deterministic
    results.
    """
    X_train, X_test, y_train, y_test, model = cfi_test_data
    cfi = CFI(
        estimator=model,
        imputation_model_continuous=LinearRegression(),
        n_permutations=20,
        method="predict",
        random_state=0,
        n_jobs=1,
    )
    cfi.fit(X_train, groups=None, var_type="auto")
    vim = cfi.importance(X_test, y_test)["importance"]
    print("reproduction")
    vim_reproducible = cfi.importance(X_test, y_test)["importance"]
    assert np.array_equal(vim, vim_reproducible)


def test_cfi_multiple_calls_are_repeatibility_None(cfi_test_data):
    """
    Test that multiple calls of .importance() when CFI is seeded provide deterministic
    results.
    """
    X_train, X_test, y_train, y_test, model = cfi_test_data
    cfi = CFI(
        estimator=model,
        imputation_model_continuous=LinearRegression(),
        n_permutations=20,
        method="predict",
        random_state=None,
        n_jobs=1,
    )
    cfi.fit(X_train, groups=None, var_type="auto")
    vim = cfi.importance(X_test, y_test)["importance"]
    # rerun importance
    vim_reproducible = cfi.importance(X_test, y_test)["importance"]
    assert np.array_equal(vim, vim_reproducible)


def test_cfi_multiple_calls_are_not_repeatibility_None(cfi_test_data):
    """
    Test that multiple calls of .importance() when CFI is seeded provide deterministic
    results.
    """
    X_train, X_test, y_train, y_test, model = cfi_test_data
    cfi = CFI(
        estimator=model,
        imputation_model_continuous=LinearRegression(),
        n_permutations=20,
        method="predict",
        random_state=None,
        n_jobs=1,
    )
    cfi.fit(X_train, groups=None, var_type="auto")
    vim = cfi.importance(X_test, y_test)["importance"]

    # refit
    cfi.fit(X_train, groups=None, var_type="auto")
    vim_reproducible = cfi.importance(X_test, y_test)["importance"]
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, vim, vim_reproducible
    )


def test_cfi_multiple_calls_are_repeatibility_rng(cfi_test_data):
    """
    Test that multiple calls of .importance() when CFI is seeded provide deterministic
    results.
    """
    X_train, X_test, y_train, y_test, model = cfi_test_data
    rng = np.random.RandomState(0)
    cfi = CFI(
        estimator=model,
        imputation_model_continuous=LinearRegression(),
        n_permutations=20,
        method="predict",
        random_state=rng,
        n_jobs=1,
    )
    cfi.fit(X_train, groups=None, var_type="auto")
    vim = cfi.importance(X_test, y_test)["importance"]
    # rerun importance
    vim_reproducible = cfi.importance(X_test, y_test)["importance"]
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, vim, vim_reproducible
    )


def test_cfi_multiple_calls_are_not_repeatibility_rng(cfi_test_data):
    """
    Test that multiple calls of .importance() when CFI is seeded provide deterministic
    results.
    """
    X_train, X_test, y_train, y_test, model = cfi_test_data
    rng = np.random.RandomState(0)
    cfi = CFI(
        estimator=model,
        imputation_model_continuous=LinearRegression(),
        n_permutations=20,
        method="predict",
        random_state=rng,
        n_jobs=1,
    )
    cfi.fit(X_train, groups=None, var_type="auto")
    vim = cfi.importance(X_test, y_test)["importance"]

    # refit
    cfi.fit(X_train, groups=None, var_type="auto")
    vim_reproducible = cfi.importance(X_test, y_test)["importance"]
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, vim, vim_reproducible
    )
