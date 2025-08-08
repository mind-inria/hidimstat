from copy import deepcopy
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier
from sklearn.cluster import FeatureAgglomeration
from sklearn.multioutput import MultiOutputClassifier

from hidimstat import PDP


def seaborn_installed():
    try:
        import seaborn

        return True
    except ImportError:
        return False


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


# test plot of the pdp
@pytest.mark.skipif(not seaborn_installed(), reason="seaborn is not installed")
@pytest.mark.mpl_image_compare
def test_plot_pdp():
    """Test simple plot of pdp"""
    import matplotlib.pyplot as plt

    seed = 42
    n_samples = 150
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
    pdp.fit_importance(X)

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(9, 8), constrained_layout=True)
    for i in range(4):
        pdp.plot(feature_id=i, X=X, ax=axs.ravel()[i])
    return fig


@pytest.mark.skipif(not seaborn_installed(), reason="seaborn is not installed")
@pytest.mark.mpl_image_compare
def test_plot_pdp_not_fig():
    """Test plot when the figure is not define before"""
    seed = 42
    n_samples = 150
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
    pdp.fit_importance(X)

    ax = pdp.plot(feature_id=0, X=X)
    return ax.get_figure()


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
        """Test importance function PDP"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(
            estimator=fitted_model,
        )
        # Test importance
        pdp.importance(X.tolist())
        assert len(pdp.importances_) > 0
        assert len(pdp.ices_) > 0
        assert pdp.pvalues_ is None

    def test_pdp_resolution_stat(self, data_generator):
        """Test resolution statistic PDP"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(estimator=fitted_model, resolution_statistique=True)
        # Test importance
        pdp.importance(X)
        assert len(pdp.importances_) > 0
        assert len(pdp.ices_) > 0
        assert pdp.pvalues_ is None

    def test_pdp_custom_values(self, data_generator):
        """Test custom values"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(estimator=fitted_model, custom_values={1: [0.1, 0.2, 0.3]})
        # Test importance
        pdp.importance(X)
        assert len(pdp.importances_) > 0
        assert len(pdp.ices_) > 0
        assert pdp.pvalues_ is None

    def test_pdp_sample_weight(self, data_generator):
        """Test sample weight"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(estimator=fitted_model, sample_weight=np.random.rand(150))
        # Test importance
        pdp.importance(X)
        assert len(pdp.importances_) > 0
        assert len(pdp.ices_) > 0
        assert pdp.pvalues_ is None

    def test_pdp_categories(self, data_generator):
        """Test categories boolean"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(estimator=fitted_model, categorical_features=np.ones(200, dtype=bool))
        # Test importance
        pdp.importance(X)
        assert len(pdp.importances_) > 0
        assert len(pdp.ices_) > 0
        assert pdp.pvalues_ is None

    def test_pdp_categories_2(self, data_generator):
        """Test categories integer"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(estimator=fitted_model, categorical_features=range(200))
        # Test importance
        pdp.importance(X)
        assert len(pdp.importances_) > 0
        assert len(pdp.ices_) > 0
        assert pdp.pvalues_ is None

    def test_pdp_features(self, data_generator):
        """Test features"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(estimator=fitted_model, features=0)
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

    def test_unknown_wrong_method_regression(self, data_generator):
        """Test when a wrong prediction method is provided"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)

        with pytest.raises(
            ValueError,
            match="The method parameter is ignored for regressors and "
            "must be 'auto' or 'predict'.",
        ):
            PDP(
                estimator=fitted_model,
                method="predict_proba",
            )

    def test_unknown_predict_method_regression(self, data_generator):
        """Test when an prediction method is not correct for regressor"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)

        with pytest.raises(
            ValueError,
            match="The method parameter is ignored for regressors and must be 'auto' or 'predict'.",
        ):
            PDP(
                estimator=fitted_model,
                method="unknown method",
            )

    def test_unknown_predict_method_classification(self, data_generator):
        """Test when an unknown prediction method is provided"""
        X, y, _, _ = data_generator
        y_clf = deepcopy(y)
        y_clf[np.where(y > 2)] = 0
        y_clf[np.where(np.logical_and(y <= 2, y > 0))] = 1
        y_clf[np.where(np.logical_and(y <= 0, y > -2))] = 2
        y_clf[np.where(y <= -2)] = 3
        y_clf = np.array(y_clf, dtype=int)
        fitted_model = RidgeClassifier().fit(X, y_clf)
        pdp = PDP(
            estimator=fitted_model,
            method="unknown method",
        )

        with pytest.raises(
            AttributeError,
            match="RidgeClassifier has none of the following attributes: unknown method",
        ):
            pdp.importance(X)

    def test_unknown_predict_method_multioutput(self, data_generator):
        """Test when an no multioutput"""
        X, y, _, _ = data_generator
        y_clf = deepcopy(y)
        y_clf[np.where(y > 2)] = 0
        y_clf[np.where(np.logical_and(y <= 2, y > 0))] = 1
        y_clf[np.where(np.logical_and(y <= 0, y > -2))] = 2
        y_clf[np.where(y <= -2)] = 3
        y_clf = np.array(y_clf, dtype=int)
        fitted_model = MultiOutputClassifier(RidgeClassifier()).fit(
            X, np.array([y_clf, y_clf]).T
        )

        with pytest.raises(
            ValueError,
            match="Multiclass-multioutput estimators are not supported",
        ):
            PDP(
                estimator=fitted_model,
            )

    def test_wrong_estimator(self, data_generator):
        """Test with wrong type of estimator"""
        X, y, _, _ = data_generator
        fitted_model = FeatureAgglomeration().fit(X, y)

        with pytest.raises(
            ValueError, match="'estimator' must be a fitted regressor or classifier."
        ):
            pdp = PDP(estimator=fitted_model)

    def test_no_importance_double(self, data_generator):
        """Test can't compute importance 2 times"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(estimator=fitted_model)
        pdp.fit_importance(X=X)

        with pytest.raises(ValueError, match="Partial Dependance Plot already fitted"):
            pdp.importance(X=X)

    def test_fit_warning(self, data_generator):
        """Test warning of methods"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(estimator=fitted_model)

        with pytest.warns(match="y won't be used"):
            pdp.fit(y=y)

        with pytest.warns(match="X won't be used"):
            pdp.fit(X=X)

        with pytest.warns(match="X won't be used"):
            with pytest.warns(match="y won't be used"):
                pdp.fit(X=X, y=y)

    def test_fit_importance_warning(self, data_generator):
        """Test warning of methods"""
        X, y, _, _ = data_generator
        pdp = PDP(estimator=LinearRegression().fit(X, y))
        with pytest.warns(match="y won't be used"):
            pdp.importance(X=X, y=y)
        pdp = PDP(estimator=LinearRegression().fit(X, y))
        with pytest.warns(match="y won't be used"):
            pdp.fit_importance(X=X, y=y)
        pdp = PDP(estimator=LinearRegression().fit(X, y))
        with pytest.warns(match="y won't be used"):
            with pytest.warns(match="cv won't be used"):
                pdp.fit_importance(X=X, y=y, cv=[])

    def test_percentage_sequence(self, data_generator):
        """Test importance method with percentage is a squence of more that 2 elements"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(estimator=fitted_model, percentiles=[2, 3, 4])

        with pytest.raises(
            ValueError, match="'percentiles' must be a sequence of 2 elements."
        ):
            pdp.importance(X=X)

    def test_percentage_no_0_1(self, data_generator):
        """Test importance method with wrong percentage"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(estimator=fitted_model, percentiles=[2, 3])

        with pytest.raises(
            ValueError, match=r"'percentiles' values must be in \[0, 1\]."
        ):
            pdp.importance(X=X)

    def test_percentage_no_right_order(self, data_generator):
        """Test importance method with wrong order for percentage"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(estimator=fitted_model, percentiles=[0.7, 0.3])

        with pytest.raises(
            ValueError,
            match=r"percentiles\[0\] must be strictly less than percentiles\[1\].",
        ):
            pdp.importance(X=X)

    def test_feature_wrong(self, data_generator):
        """Test importance method with wrong grid resolution"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(estimator=fitted_model, features=[0, 1, 3, -10])

        with pytest.raises(ValueError, match=r"all features must be in \[0, "):
            pdp.importance(X=X)

    def test_empty_categorical_features(self, data_generator):
        """Test empty list for categorical features"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(estimator=fitted_model, categorical_features=[])

        with pytest.raises(
            ValueError,
            match="supported. Use `None` instead to indicate that there are no ",
        ):
            pdp.importance(X=X)

    def test_empty_categorical_features_binary(self, data_generator):
        """Test missing feature for categories in binary selection"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(estimator=fitted_model, categorical_features=np.ones(50, dtype=bool))

        with pytest.raises(
            ValueError,
            match="When `categorical_features` is a boolean array-like, ",
        ):
            pdp.importance(X=X)

    def test_empty_categorical_features_wrong_type(self, data_generator):
        """Test categorical feature is wrong type"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(
            estimator=fitted_model, categorical_features=np.ones(200, dtype=float)
        )

        with pytest.raises(
            ValueError,
            match="Expected `categorical_features` to be an array-like of boolean, ",
        ):
            pdp.importance(X=X)

    def test_grid(self, data_generator):
        """Test importance method with wrong grid resolution"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(estimator=fitted_model, grid_resolution=-10)

        with pytest.raises(
            ValueError, match="'grid_resolution' must be strictly greater than 1."
        ):
            pdp.importance(X=X)

    def test_percentiles_close(self, data_generator):
        """Test importance method with closest percentile"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(estimator=fitted_model, percentiles=[0.1, 0.100001])

        with pytest.raises(
            ValueError, match="percentiles are too close to each other,"
        ):
            pdp.importance(X=X)

    def test_wrong_custom_value(self, data_generator):
        """Test high dimention of custom values"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(estimator=fitted_model, custom_values={1: [[1, 2, 3], [4, 5, 6]]})

        with pytest.raises(
            ValueError,
            match="The custom grid for some features is not a one-dimensional array.",
        ):
            pdp.importance(X=X)

    def test_wrong_type(self, data_generator):
        """Test wrong type"""
        X = np.array(["A", "B", "C", np.nan], dtype=object).reshape(-1, 1)
        y = np.array([0, 1, 0, 1])

        from sklearn.preprocessing import OrdinalEncoder
        from sklearn.pipeline import make_pipeline

        fitted_model = make_pipeline(
            OrdinalEncoder(encoded_missing_value=-1),
            LogisticRegression(),
        ).fit(X, y)

        pdp = PDP(
            estimator=fitted_model,
        )

        with pytest.raises(
            ValueError, match="Finding unique categories fail due to sorting."
        ):
            pdp.importance(X=X)

    def test_reject_interger_data(self, data_generator):
        """integer should be categorical"""
        X = np.arange(8).reshape(4, 2)
        y = np.array([0, 1, 0, 1])
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(
            estimator=fitted_model,
        )
        with pytest.raises(
            ValueError,
            match="Partial dependence plots are not supported for integer data: this ",
        ):
            pdp.importance(X=X)

    def test_unfitted_importance(self, data_generator):
        """Test plot before fitting"""
        X, y, _, _ = data_generator
        fitted_model = LinearRegression().fit(X, y)
        pdp = PDP(
            estimator=fitted_model,
        )

        with pytest.raises(
            ValueError, match="The importances need to be called before."
        ):
            pdp.plot(feature_id=0, X=X)
