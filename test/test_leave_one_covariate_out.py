# import numpy as np
# import pandas as pd
# import pytest
# from sklearn.exceptions import NotFittedError
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.metrics import log_loss
# from sklearn.model_selection import train_test_split
# from hidimstat._utils.scenario import multivariate_simulation

# from hidimstat import loco, LOCO
# from hidimstat.base_perturbation import BasePerturbation


# def test_loco():
#     """Test the Leave-One-Covariate-Out algorithm on a linear scenario."""
#     X, y, beta, noise = multivariate_simulation(
#         n_samples=150,
#         n_features=200,
#         support_size=10,
#         shuffle=False,
#         seed=42,
#     )
#     important_features = np.where(beta != 0)[0]
#     non_important_features = np.where(beta == 0)[0]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#     regression_model = LinearRegression()
#     regression_model.fit(X_train, y_train)

#     loco = LOCO(
#         estimator=regression_model,
#         method="predict",
#         n_jobs=1,
#     )

#     loco.fit(
#         X_train,
#         y_train,
#         groups=None,
#     )
#     importance = loco.importance(X_test, y_test)

#     assert importance.shape == (X.shape[1],)
#     assert (
#         importance[important_features].mean()
#         > importance[non_important_features].mean()
#     )

#     # Same with groups and a pd.DataFrame
#     groups = {
#         "group_0": [f"col_{i}" for i in important_features],
#         "the_group_1": [f"col_{i}" for i in non_important_features],
#     }
#     X_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
#     X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, random_state=0)
#     regression_model.fit(X_train_df, y_train)
#     loco = LOCO(
#         estimator=regression_model,
#         method="predict",
#         n_jobs=1,
#     )
#     loco.fit(
#         X_train_df,
#         y_train,
#         groups=groups,
#     )
#     # warnings because we doesn't considere the name of columns of pandas
#     with pytest.warns(UserWarning, match="X does not have valid feature names, but"):
#         importance = loco.importance(X_test_df, y_test)

#     assert importance[0].mean() > importance[1].mean()

#     # Classification case
#     y_clf = np.where(y > np.median(y), 1, 0)
#     _, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, random_state=0)
#     logistic_model = LogisticRegression()
#     logistic_model.fit(X_train, y_train_clf)

#     loco_clf = LOCO(
#         estimator=logistic_model,
#         method="predict_proba",
#         n_jobs=1,
#         loss=log_loss,
#     )
#     loco_clf.fit(
#         X_train,
#         y_train_clf,
#         groups={"group_0": important_features, "the_group_1": non_important_features},
#     )
#     importance_clf = loco_clf.importance(X_test, y_test_clf)

#     assert importance_clf.shape == (2,)
#     assert importance[0].mean() > importance[1].mean()


# def test_raises_value_error():
#     """Test for error when model does not have predict_proba or predict."""
#     X, y, beta, noise = multivariate_simulation(
#         n_samples=150,
#         n_features=200,
#         support_size=10,
#         shuffle=False,
#         seed=42,
#     )
#     # Not fitted estimator
#     with pytest.raises(NotFittedError):
#         loco = LOCO(
#             estimator=LinearRegression(),
#             method="predict",
#         )

#     # Not fitted sub-model when calling importance and predict
#     with pytest.raises(ValueError, match="The class is not fitted."):
#         fitted_model = LinearRegression().fit(X, y)
#         loco = LOCO(
#             estimator=fitted_model,
#             method="predict",
#         )
#         loco.predict(X)
#     with pytest.raises(ValueError, match="The class is not fitted."):
#         fitted_model = LinearRegression().fit(X, y)
#         loco = LOCO(
#             estimator=fitted_model,
#             method="predict",
#         )
#         loco.importance(X, y)

#     with pytest.raises(
#         ValueError, match="The estimators require to be fit before to use them"
#     ):
#         fitted_model = LinearRegression().fit(X, y)
#         loco = LOCO(
#             estimator=fitted_model,
#             method="predict",
#         )
#         BasePerturbation.fit(loco, X, y)
#         loco.importance(X, y)


# def test_loco_function():
#     """Test the function of LOCO algorithm on a linear scenario."""
#     X, y, beta, noise = multivariate_simulation(
#         n_samples=150,
#         n_features=200,
#         support_size=10,
#         shuffle=False,
#         seed=42,
#     )
#     important_features = np.where(beta != 0)[0]
#     non_important_features = np.where(beta == 0)[0]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#     regression_model = LinearRegression()
#     regression_model.fit(X_train, y_train)

#     selection, importance, pvalue = loco(
#         regression_model,
#         X,
#         y,
#         method="predict",
#         n_jobs=1,
#     )

#     assert importance.shape == (X.shape[1],)
#     assert (
#         importance[important_features].mean()
#         > importance[non_important_features].mean()
#     )


from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import RidgeCV, LogisticRegressionCV

from hidimstat import loco, LOCO
from hidimstat.base_perturbation import BasePerturbation
from hidimstat._utils.exception import InternalError


def run_loco(X, y):
    """
    Configure Leave One Covariate Out (LOCO) model with linear regression
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
    3. Configures LOCO with linear regression as both estimator and imputer
    4. Calculates feature importance using the test set
    The LOCO method uses permutation-based importance scoring with linear
    regression as the base model.
    """
    # split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # create and fit a linear regression model on the training set
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    # instantiate LOCO model with linear regression imputer
    loco = LOCO(
        estimator=regression_model,
        method="predict",
        n_jobs=1,
    )
    # fit the model using the training set
    loco.fit(
        X_train,
        y_train,
        groups=None,
    )
    # calculate feature importance using the test set
    importance = loco.importance(X_test, y_test)
    return importance


##############################################################################
## tests loco on different type of data
parameter_exact = [
    ("HiDim", 150, 200, 10, 0.0, 42, 1.0, np.inf, 0.0),
    ("HiDim with correlated features", 150, 200, 10, 0.2, 42, 1.0, np.inf, 0.0),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_exact))[1:])),
    ids=list(zip(*parameter_exact))[0],
)
def test_linear_data_exact(data_generator):
    """Tests the method on linear cases with noise and correlation"""
    X, y, important_features, _ = data_generator
    importance = run_loco(X, y)
    # check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)
    # check that important features have the highest importance scores
    assert np.all([int(i) in important_features for i in np.argsort(importance)[-10:]])


parameter_partial = [
    ("HiDim with noise", 150, 200, 10, 0.0, 42, 1.0, 10.0, 0.0),
    ("HiDim with correlated noise", 150, 200, 10, 0.0, 42, 1.0, 10.0, 0.2),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_partial))[1:])),
    ids=list(zip(*parameter_partial))[0],
)
def test_linear_data_partial(data_generator, rho):
    """Tests the method on linear cases with noise and correlation"""
    X, y, important_features, _ = data_generator
    importance = run_loco(X, y)
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


parameter_fail = [
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
    ("high level noise", 150, 200, 10, 0.2, 42, 1.0, 1.0, 0.0),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_fail))[1:])),
    ids=list(zip(*parameter_fail))[0],
)
def test_linear_data_fail(data_generator):
    """Tests when the method doesn't identify all important features"""
    X, y, important_features, not_important_features = data_generator
    importance = run_loco(X, y)
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
## Test specific options of loco
@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(150, 200, 10, 0.0, 42, 1.0, np.inf, 0.0)],
    ids=["high dimension"],
)
def test_group(data_generator):
    """Test LOCO with groups using pandas objects"""
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

    loco = LOCO(
        estimator=regression_model,
        method="predict",
        n_jobs=1,
    )
    loco.fit(
        X_train_df,
        y_train,
        groups=groups,
    )
    # Warning expected since column names in pandas are not considered
    with pytest.warns(UserWarning, match="X does not have valid feature names, but"):
        importance = loco.importance(X_test_df, y_test)

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
    """Test LOCO for a classification problem"""
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

    loco = LOCO(
        estimator=logistic_model,
        n_jobs=1,
        method="predict_proba",
        loss=log_loss,
    )
    loco.fit(
        X_train,
        y_train_clf,
        groups=None,
    )
    importance = loco.importance(X_test, y_test_clf)
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
class TestLOCOClass:
    """Test the element of the class"""

    def test_init(self, data_generator):
        """Test LOCO initialization"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        loco = LOCO(
            estimator=fitted_model,
            method="predict",
        )
        assert loco.n_jobs == 1
        assert loco.n_permutations == 1
        assert loco.loss == root_mean_squared_error
        assert loco.method == "predict"

    def test_fit(self, data_generator):
        """Test fitting LOCO"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        loco = LOCO(estimator=fitted_model)

        # Test fit with specified groups
        groups = {"g1": [0, 1], "g2": [2, 3, 4]}
        loco.fit(X, y, groups=groups)
        assert loco._n_groups == 2

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
        """Test LOCO with categorical variables"""
        rng = np.random.default_rng(seed)
        X_cont = rng.random((n_samples, 2))
        X_cat = rng.integers(low=0, high=3, size=(n_samples, 1))
        X = np.hstack([X_cont, X_cat])
        y = rng.random((n_samples, 1))
        fitted_model = LinearRegression().fit(X, y)

        loco = LOCO(estimator=fitted_model)

        loco.fit(X, y)

        importances = loco.importance(X, y)
        assert len(importances) == 3
        assert np.all(importances >= 0)


##############################################################################
@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(150, 200, 10, 0.0, 42, 1.0, 0.0, 0.0)],
    ids=["default data"],
)
class TestLOCOExceptions:
    """Test class for LOCO exceptions"""

    def test_unfitted_estimator(self, data_generator):
        """Test when using an unfitted estimator"""
        with pytest.raises(NotFittedError):
            LOCO(
                estimator=LinearRegression(),
                method="predict",
            )

    def test_unknown_predict_method(self, data_generator):
        """Test when an unknown prediction method is provided"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)

        with pytest.raises(ValueError):
            LOCO(
                estimator=fitted_model,
                method="unknown method",
            )

    def test_unfitted_predict(self, data_generator):
        """Test predict method with unfitted model"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        loco = LOCO(
            estimator=fitted_model,
            method="predict",
        )

        with pytest.raises(ValueError, match="The class is not fitted."):
            loco.predict(X)

    def test_unfitted_importance(self, data_generator):
        """Test importance method with unfitted model"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        loco = LOCO(
            estimator=fitted_model,
            method="predict",
        )

        with pytest.raises(ValueError, match="The class is not fitted."):
            loco.importance(X, y)

    def test_not_good_type_X(self, data_generator):
        """Test when X is wrong type"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        loco = LOCO(
            estimator=fitted_model,
            method="predict",
        )
        loco.fit(X, y, groups=None)

        with pytest.raises(
            ValueError, match="X should be a pandas dataframe or a numpy array."
        ):
            loco.importance(X.tolist(), y)

    def test_mismatched_features(self, data_generator):
        """Test when number of features doesn't match between fit and predict"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        loco = LOCO(
            estimator=fitted_model,
            method="predict",
        )
        loco.fit(X, y, groups=None)

        with pytest.raises(
            AssertionError, match="X does not correspond to the fitting data."
        ):
            loco.importance(X[:, :-1], y)

    def test_mismatched_features_string(self, data_generator):
        """Test when name of features doesn't match between fit and predict"""
        X, y, _, _ = data_generator
        X = pd.DataFrame({"col_" + str(i): X[:, i] for i in range(X.shape[1])})
        fitted_model = LinearRegression().fit(X, y)
        loco = LOCO(
            estimator=fitted_model,
            method="predict",
        )
        subgroups = {
            "group1": ["col_" + str(i) for i in range(int(X.shape[1] / 2))],
            "group2": [
                "col_" + str(i) for i in range(int(X.shape[1] / 2), X.shape[1] - 3)
            ],
        }
        loco.fit(X, y, groups=subgroups)

        with pytest.raises(
            AssertionError,
            match=f"The array is missing at least one of the following columns \['col_100', 'col_101', 'col_102',",
        ):
            loco.importance(
                X[np.concatenate([subgroups["group1"], subgroups["group2"][:-2]])], y
            )

    def test_internal_error(self, data_generator):
        """Test when name of features doesn't match between fit and predict"""
        X, y, _, _ = data_generator
        X = pd.DataFrame({"col_" + str(i): X[:, i] for i in range(X.shape[1])})
        fitted_model = LinearRegression().fit(X, y)
        loco = LOCO(
            estimator=fitted_model,
            method="predict",
        )
        subgroups = {
            "group1": ["col_" + str(i) for i in range(int(X.shape[1] / 2))],
            "group2": [
                "col_" + str(i) for i in range(int(X.shape[1] / 2), X.shape[1] - 3)
            ],
        }
        loco.fit(X, y, groups=subgroups)
        loco.features_groups["group1"] = [None for i in range(100)]

        X = X.to_records(index=False)
        X = np.array(X, dtype=X.dtype.descr)
        with pytest.raises(
            InternalError,
            match=f"A problem with indexing has happened during the fit.",
        ):
            loco.importance(X, y)

    def test_invalid_groups_format(self, data_generator):
        """Test when groups are provided in invalid format"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        loco = LOCO(estimator=fitted_model, method="predict")

        invalid_groups = ["group1", "group2"]  # Should be dictionary
        with pytest.raises(ValueError, match="groups needs to be a dictionnary"):
            loco.fit(X, y, groups=invalid_groups)

    def test_groups_warning(self, data_generator):
        """Test if a subgroup raise a warning"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        loco = LOCO(
            estimator=fitted_model,
            method="predict",
        )
        subgroups = {"group1": [0, 1], "group2": [2, 3]}
        loco.fit(X, y, groups=subgroups)

        with pytest.warns(
            UserWarning,
            match="The number of features in X: 200 differs from the"
            " number of features for which importance is computed: 4",
        ):
            loco.importance(X, y)


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(150, 200, 10, 0.2, 42, 1.0, 1.0, 0.0)],
    ids=["high level noise"],
)
def test_function_loco(data_generator):
    """Test LOCO function"""
    X, y, _, _ = data_generator
    loco(
        LinearRegression().fit(X, y),
        X,
        y,
    )
