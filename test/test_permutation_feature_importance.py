from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

from hidimstat import pfi, PFI
from hidimstat._utils.exception import InternalError


def run_pfi(X, y, n_permutation, seed):
    """
    Configure Permutation Feature Importance (PFI) model with linear regression
    for feature importance analysis.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix where each column represents a feature
        and each row a sample.
    y : array-like of shape (n_samples,)
        Target variable array.
    n_permutation : int
        Number of permutations to perform for the PFI analysis.
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
    3. Configures PFI with linear regression as both estimator and imputer
    4. Calculates feature importance using the test set
    The PFI method uses permutation-based importance scoring with linear
    regression as the base model.
    """
    # split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # create and fit a linear regression model on the training set
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    # instantiate PFI model with linear regression imputer
    pfi = PFI(
        estimator=regression_model,
        n_permutations=n_permutation,
        method="predict",
        random_state=seed,
        n_jobs=1,
    )
    # fit the model using the training set
    pfi.fit(
        X_train,
        groups=None,
    )
    # calculate feature importance using the test set
    importance = pfi.importance(X_test, y_test)
    return importance


##############################################################################
## tests pfi on different type of data
parameter_exact = [
    ("HiDim", 150, 200, 10, 0.0, 42, 1.0, np.inf, 0.0),
    ("HiDim with noise", 150, 200, 10, 0.0, 42, 1.0, 10.0, 0.0),
    ("HiDim with correlated noise", 150, 200, 10, 0.0, 42, 1.0, 10.0, 0.2),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_exact))[1:])),
    ids=list(zip(*parameter_exact))[0],
)
@pytest.mark.parametrize("n_permutation, pfi_seed", [(10, 0)], ids=["default_pfi"])
def test_linear_data_exact(data_generator, n_permutation, pfi_seed):
    """Tests the method on linear cases with noise and correlation"""
    X, y, important_features, _ = data_generator
    importance = run_pfi(X, y, n_permutation, pfi_seed)
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
@pytest.mark.parametrize("n_permutation, pfi_seed", [(10, 0)], ids=["default_pfi"])
def test_linear_data_partial(data_generator, n_permutation, pfi_seed):
    """Tests the method on linear cases with noise and correlation"""
    X, y, important_features, _ = data_generator
    importance = run_pfi(X, y, n_permutation, pfi_seed)
    # check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)
    # check that important features have the highest importance scores
    min_rank = 0
    importance_sort = np.flip(np.argsort(importance))
    for index in important_features:
        rank = np.where(importance_sort == index)[0]
        if rank > min_rank:
            min_rank = rank
    # accept missing ranking of 20 elements
    assert min_rank < 30


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(150, 200, 10, 0.2, 42, 1.0, np.inf, 0.0), (150, 200, 10, 0.2, 42, 1.0, 1.0, 0.0)],
    ids=["HiDim with correlated features", "high level noise"],
)
@pytest.mark.parametrize("n_permutation, pfi_seed", [(20, 0)], ids=["default_pfi"])
def test_linear_data_fail(data_generator, n_permutation, pfi_seed):
    """Tests when the method doesn't identify all important features"""
    X, y, important_features, not_important_features = data_generator
    importance = run_pfi(X, y, n_permutation, pfi_seed)
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
## Test specific options of pfi
@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(150, 200, 10, 0.0, 42, 1.0, np.inf, 0.0)],
    ids=["high dimension"],
)
def test_group(data_generator):
    """Test PFI with groups using pandas objects"""
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

    pfi = PFI(
        estimator=regression_model,
        n_permutations=20,
        method="predict",
        random_state=0,
        n_jobs=1,
    )
    pfi.fit(
        X_train_df,
        groups=groups,
    )
    # Warning expected since column names in pandas are not considered
    with pytest.warns(UserWarning, match="X does not have valid feature names, but"):
        importance = pfi.importance(X_test_df, y_test)

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
    """Test PFI for a classification problem"""
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

    pfi = PFI(
        estimator=logistic_model,
        n_permutations=20,
        random_state=0,
        n_jobs=1,
        method="predict_proba",
        loss=log_loss,
    )
    pfi.fit(
        X_train,
        groups=None,
    )
    importance = pfi.importance(X_test, y_test_clf)
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
class TestPFIClass:
    """Test the element of the class"""

    def test_init(self, data_generator):
        """Test PFI initialization"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pfi = PFI(
            estimator=fitted_model,
            method="predict",
        )
        assert pfi.n_jobs == 1
        assert pfi.n_permutations == 50
        assert pfi.loss == root_mean_squared_error
        assert pfi.method == "predict"

    def test_fit(self, data_generator):
        """Test fitting PFI"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pfi = PFI(
            estimator=fitted_model,
            random_state=42,
        )

        # Test fit with specified groups
        groups = {"g1": [0, 1], "g2": [2, 3, 4]}
        pfi.fit(X, groups=groups)
        assert pfi._n_groups == 2

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
        """Test PFI with categorical variables"""
        rng = np.random.default_rng(seed)
        X_cont = rng.random((n_samples, 2))
        X_cat = rng.integers(low=0, high=3, size=(n_samples, 1))
        X = np.hstack([X_cont, X_cat])
        y = rng.random((n_samples, 1))
        fitted_model = LinearRegression().fit(X, y)

        pfi = PFI(
            estimator=fitted_model,
            random_state=seed + 1,
        )

        pfi.fit(X, y)

        importances = pfi.importance(X, y)
        assert len(importances) == 3
        assert np.all(importances >= 0)


##############################################################################
@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(150, 200, 10, 0.0, 42, 1.0, 0.0, 0.0)],
    ids=["default data"],
)
class TestPFIExceptions:
    """Test class for PFI exceptions"""

    def test_unfitted_estimator(self, data_generator):
        """Test when using an unfitted estimator"""
        with pytest.raises(NotFittedError):
            PFI(
                estimator=LinearRegression(),
                method="predict",
            )

    def test_unknown_predict_method(self, data_generator):
        """Test when an unknown prediction method is provided"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)

        with pytest.raises(ValueError):
            PFI(
                estimator=fitted_model,
                method="unknown method",
            )

    def test_unfitted_predict(self, data_generator):
        """Test predict method with unfitted model"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pfi = PFI(
            estimator=fitted_model,
            method="predict",
        )

        with pytest.raises(ValueError, match="The class is not fitted."):
            pfi.predict(X)

    def test_unfitted_importance(self, data_generator):
        """Test importance method with unfitted model"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pfi = PFI(
            estimator=fitted_model,
            method="predict",
        )

        with pytest.raises(ValueError, match="The class is not fitted."):
            pfi.importance(X, y)

        """Test when invalid number of permutations is provided"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)

        with pytest.raises(AssertionError, match="n_permutations must be positive"):
            PFI(estimator=fitted_model, n_permutations=-1, method="predict")

    def test_not_good_type_X(self, data_generator):
        """Test when X is wrong type"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pfi = PFI(
            estimator=fitted_model,
            method="predict",
        )
        pfi.fit(X, groups=None)

        with pytest.raises(
            ValueError, match="X should be a pandas dataframe or a numpy array."
        ):
            pfi.importance(X.tolist(), y)

    def test_mismatched_features(self, data_generator):
        """Test when number of features doesn't match between fit and predict"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pfi = PFI(
            estimator=fitted_model,
            method="predict",
        )
        pfi.fit(X, groups=None)

        with pytest.raises(
            AssertionError, match="X does not correspond to the fitting data."
        ):
            pfi.importance(X[:, :-1], y)

    def test_mismatched_features_string(self, data_generator):
        """Test when name of features doesn't match between fit and predict"""
        X, y, _, _ = data_generator
        X = pd.DataFrame({"col_" + str(i): X[:, i] for i in range(X.shape[1])})
        fitted_model = LinearRegression().fit(X, y)
        pfi = PFI(
            estimator=fitted_model,
            method="predict",
        )
        subgroups = {
            "group1": ["col_" + str(i) for i in range(int(X.shape[1] / 2))],
            "group2": [
                "col_" + str(i) for i in range(int(X.shape[1] / 2), X.shape[1] - 3)
            ],
        }
        pfi.fit(X, groups=subgroups)

        with pytest.raises(
            AssertionError,
            match=f"The array is missing at least one of the following columns \['col_100', 'col_101', 'col_102',",
        ):
            pfi.importance(
                X[np.concatenate([subgroups["group1"], subgroups["group2"][:-2]])], y
            )

    def test_internal_error(self, data_generator):
        """Test when name of features doesn't match between fit and predict"""
        X, y, _, _ = data_generator
        X = pd.DataFrame({"col_" + str(i): X[:, i] for i in range(X.shape[1])})
        fitted_model = LinearRegression().fit(X, y)
        pfi = PFI(
            estimator=fitted_model,
            method="predict",
        )
        subgroups = {
            "group1": ["col_" + str(i) for i in range(int(X.shape[1] / 2))],
            "group2": [
                "col_" + str(i) for i in range(int(X.shape[1] / 2), X.shape[1] - 3)
            ],
        }
        pfi.fit(X, groups=subgroups)
        pfi.features_groups["group1"] = [None for i in range(100)]

        X = X.to_records(index=False)
        X = np.array(X, dtype=X.dtype.descr)
        with pytest.raises(
            InternalError,
            match=f"A problem with indexing has happened during the fit.",
        ):
            pfi.importance(X, y)

    def test_invalid_groups_format(self, data_generator):
        """Test when groups are provided in invalid format"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pfi = PFI(estimator=fitted_model, method="predict")

        invalid_groups = ["group1", "group2"]  # Should be dictionary
        with pytest.raises(ValueError, match="groups needs to be a dictionnary"):
            pfi.fit(X, groups=invalid_groups)

    def test_groups_warning(self, data_generator):
        """Test if a subgroup raise a warning"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pfi = PFI(
            estimator=fitted_model,
            method="predict",
        )
        subgroups = {"group1": [0, 1], "group2": [2, 3]}
        pfi.fit(X, y, groups=subgroups)

        with pytest.warns(
            UserWarning,
            match="The number of features in X: 200 differs from the"
            " number of features for which importance is computed: 4",
        ):
            pfi.importance(X, y)


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(150, 200, 10, 0.2, 42, 1.0, 1.0, 0.0)],
    ids=["high level noise"],
)
@pytest.mark.parametrize("n_permutation, pfi_seed", [(20, 0)], ids=["default_pfi"])
def test_function_pfi(data_generator, n_permutation, pfi_seed):
    """Test PFI function"""
    X, y, _, _ = data_generator
    pfi(
        LinearRegression().fit(X, y),
        X,
        y,
        n_permutations=n_permutation,
        random_state=pfi_seed,
    )
