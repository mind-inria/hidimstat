import numpy as np
import pytest

from sklearn.covariance import LedoitWolf, GraphicalLassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold

from hidimstat import CRT
from hidimstat.conditional_randomization_test import crt
from hidimstat.statistical_tools.gaussian_knockoff import GaussianGenerator
from hidimstat.statistical_tools.multiple_testing import fdp_power
from hidimstat.statistical_tools.lasso_test import lasso_statistic


def configure_linear_categorial_crt(X, y, n_repeat, seed, fdr):
    """
    Configure Conditional Randomize Test (CRT) model with linear regression
    for feature importance analysis.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix where each column represents a feature
        and each row a sample.
    y : array-like of shape (n_samples,)
        Target variable array.
    n_repeat : int
        Number of permutations to perform for the CRT analysis.
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
    1. Configures CRT
    2. Calculates feature importance
    The CRT method uses permutation-based importance scoring with linear
    regression as the base model.
    """
    # instantiate CRT model with linear regression imputer
    crt = CRT(
        n_repeat=n_repeat,
        generator=GaussianGenerator(
            cov_estimator=LedoitWolf(assume_centered=True), random_state=seed
        ),
    )
    # fit the model using the training set
    importance = crt.fit_importance(X, y)
    selected = crt.selection_fdr(fdr=fdr)
    return importance, selected


##############################################################################
## tests crt on different type of data
parameter_exact = [
    ("Dim", 300, 20, 5, 0.0, 42, 1.0, np.inf, 0.0),
    ("Dim with noise", 300, 20, 5, 0.0, 42, 1.0, 10.0, 0.0),
    ("Dim with correlated noise", 300, 20, 5, 0.0, 42, 1.0, 10.0, 0.2),
    ("Dim with correlated features", 300, 20, 5, 0.2, 42, 1.0, np.inf, 0.0),
    ("Dim high level noise", 300, 20, 5, 0.2, 42, 1.0, 1.0, 0.0),
    ("Dim with correlated features and noise", 300, 20, 5, 0.2, 42, 1, 10, 0),
    (
        "Dim with correlated features and correlated noise",
        300,
        20,
        5,
        0.2,
        42,
        1.0,
        10,
        0.2,
    ),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_exact))[1:])),
    ids=list(zip(*parameter_exact))[0],
)
@pytest.mark.parametrize(
    "crt_n_sampling, crt_seed, crt_fdr", [(5, 5, 0.4)], ids=["default_crt"]
)
def test_crt_linear_data_exact(data_generator, crt_n_sampling, crt_seed, crt_fdr):
    """Tests the method on linear cases with noise and correlation"""
    X, y, important_features, _ = data_generator
    importance, selected = configure_linear_categorial_crt(
        X, y, crt_n_sampling, crt_seed, crt_fdr
    )
    # check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)
    # check that important features have the highest importance scores
    assert np.all([int(i) in important_features for i in np.argsort(importance)[-5:]])

    # test the selection part
    fdp, power = fdp_power(np.where(selected)[0], important_features)
    assert fdp < crt_fdr
    assert power == 1.0


## tests crt no power
parameter_no_power = [
    ("Dim", 200, 20, 3, 0.0, 42, 1.0, np.inf, 0.0),
    ("Dim with noise", 200, 20, 3, 0.0, 42, 1.0, 10.0, 0.0),
    ("Dim with correlated noise", 200, 20, 3, 0.0, 42, 1.0, 10.0, 0.2),
    ("Dim with correlated features", 200, 20, 3, 0.2, 42, 1.0, np.inf, 0.0),
    ("Dim high level noise", 200, 20, 3, 0.2, 42, 1.0, 1.0, 0.0),
    ("Dim with correlated features and noise", 200, 20, 3, 0.2, 42, 1, 10, 0),
    (
        "Dim with correlated features and correlated noise",
        200,
        20,
        3,
        0.2,
        42,
        1.0,
        10,
        0.2,
    ),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_no_power))[1:])),
    ids=list(zip(*parameter_no_power))[0],
)
@pytest.mark.parametrize(
    "crt_n_sampling, crt_seed, crt_fdr", [(5, 5, 0.4)], ids=["default_crt"]
)
def test_crt_linear_no_power(data_generator, crt_n_sampling, crt_seed, crt_fdr):
    """Tests the method on linear cases with noise and correlation"""
    X, y, important_features, _ = data_generator
    importance, selected = configure_linear_categorial_crt(
        X, y, crt_n_sampling, crt_seed, crt_fdr
    )
    # check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)
    # check that important features have the highest importance scores
    assert np.all([int(i) in important_features for i in np.argsort(importance)[-3:]])

    # test the selection part
    fdp, power = fdp_power(np.where(selected)[0], important_features)
    assert fdp < crt_fdr
    assert power == 0.0


## tests crt on different type of data
parameter_bad_detection = [
    ("No information", 300, 20, 5, 0.0, 42, 1.0, 0.0, 0.0),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_bad_detection))[1:])),
    ids=list(zip(*parameter_bad_detection))[0],
)
@pytest.mark.parametrize(
    "crt_n_sampling, crt_seed, crt_fdr", [(5, 5, 0.4)], ids=["default_crt"]
)
def test_crt_linear_fail(data_generator, crt_n_sampling, crt_seed, crt_fdr):
    """Tests the method on linear cases with noise and correlation"""
    X, y, important_features, _ = data_generator
    importance, selected = configure_linear_categorial_crt(
        X, y, crt_n_sampling, crt_seed, crt_fdr
    )
    # check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)
    # Verify that not all important features are detected
    assert np.sum(
        [int(i) in important_features for i in np.argsort(importance)[-4:]]
    ) != len(important_features)

    # test the selection part
    fdp, power = fdp_power(np.where(selected)[0], important_features)
    assert fdp == 0.0
    assert power == 0.0


##############################################################################
@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(300, 20, 5, 0.0, 42, 1.0, np.inf, 0.0)],
    ids=["default data"],
)
class TestCRTClass:
    """Test the element of the class"""

    def test_crt_init(self, data_generator):
        """Test CRT initialization"""
        X, y, _, _ = data_generator
        crt = CRT()
        assert crt.n_jobs == 1
        assert crt.n_repeat == 10
        assert crt.generator is not None
        assert crt.statistical_test is not None

    def test_crt_fit(self, data_generator):
        """Test fitting CRT"""
        X, y, _, _ = data_generator
        crt = CRT()
        crt.fit(X)
        # check if the generator is fitted
        crt.generator._check_fit()

    def test_crt_importance(self, data_generator):
        """Test importance of CRT"""
        X, y, important_features, _ = data_generator
        crt = CRT(n_repeat=5)
        crt.fit(X)
        importance = crt.importance(X, y)

        # check that importance scores are defined for each feature
        assert importance.shape == (X.shape[1],)
        # check that important features have the highest importance scores
        assert np.all(
            [int(i) in important_features for i in np.argsort(importance)[-5:]]
        )

    def test_crt_categorical(
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
        """Test CRT with categorical variables"""
        rng = np.random.default_rng(seed)
        X_cont = rng.random((n_samples, 2))
        X_cat = rng.integers(low=0, high=3, size=(n_samples, 1))
        X = np.hstack([X_cont, X_cat])
        y = rng.random((n_samples))

        crt = CRT()

        with pytest.warns(
            Warning,
            match="The conditional covariance matrix for knockoffs is not positive definite.",
        ):
            with pytest.warns(
                Warning,
                match="The equi-correlated matrix for knockoffs is not positive definite.",
            ):
                importances = crt.fit_importance(X, y)
        assert len(importances) == 3
        assert np.all(importances >= 0)

    def test_crt_CV_estimator(self, data_generator, seed):
        """Test CRT with a crossvalidation estimator"""
        fdr = 0.5
        X, y, important_features, _ = data_generator

        def lasso_statistic_gen(X, y):
            return lasso_statistic(
                X,
                y,
                lasso=GridSearchCV(
                    Lasso(), param_grid={"alpha": np.linspace(0.2, 0.3, 5)}
                ),
            )

        crt = CRT(
            n_repeat=1,
            generator=GaussianGenerator(
                cov_estimator=LedoitWolf(assume_centered=True), random_state=seed + 2
            ),
            statistical_test=lasso_statistic_gen,
        )
        crt.fit_importance(X, y)
        selected = crt.selection_fdr(fdr=fdr)

        fdp, power = fdp_power(np.where(selected)[0], important_features)

        assert fdp <= fdr
        assert power > 0.7

    def test_estimate_distribution(self, data_generator, seed):
        """Test different estimation of the covariance"""
        fdr = 0.2
        X, y, important_features, _ = data_generator
        crt = CRT(
            n_repeat=1,
            generator=GaussianGenerator(
                cov_estimator=LedoitWolf(assume_centered=True), random_state=seed + 1
            ),
        )
        crt.fit_importance(X, y)
        selected = crt.selection_fdr(fdr=fdr)
        for i in np.where(selected)[0]:
            assert np.any(i == important_features)

        crt = CRT(
            n_repeat=1,
            generator=GaussianGenerator(
                cov_estimator=GraphicalLassoCV(
                    alphas=[1e-3, 1e-2, 1e-1, 1],
                    cv=KFold(n_splits=5, shuffle=True, random_state=0),
                ),
                random_state=seed + 2,
            ),
        )
        crt.fit_importance(X, y)
        selected = crt.selection_fdr(fdr=fdr)
        for i in np.where(selected)[0]:
            assert np.any(i == important_features)

    def test_crt_selection(self, data_generator):
        """Test the selection of variable from knockoff"""
        fdr = 0.5
        n_repeat = 5
        X, y, important_features, _ = data_generator
        crt = CRT(n_repeat=n_repeat)
        crt.fit_importance(X, y)
        selected = crt.selection_fdr(fdr=fdr)

        fdp, power = fdp_power(np.where(selected)[0], important_features)
        assert fdp <= 0.2
        assert power > 0.7
        assert np.all(0 <= crt.pvalues_) or np.all(crt.pvalues_ <= 1)

    def test_crt_repeat_quantile(self, data_generator, n_features):
        """Test crt selection"""
        fdr = 0.5
        n_repeat = 5
        X, y, important_features, _ = data_generator
        crt = CRT(n_repeat=n_repeat)
        crt.fit_importance(X, y)
        selected = crt.selection_fdr(fdr=fdr)

        fdp, power = fdp_power(np.where(selected)[0], important_features)

        assert crt.test_scores_.shape == (n_repeat, n_features)
        assert fdp < 0.5
        assert power == 1.0

    def test_crt_repeat_e_values(self, data_generator, n_features):
        """Test crt selection with e-values"""
        fdr = 0.5
        n_repeat = 5
        X, y, important_features, _ = data_generator
        crt = CRT(n_repeat=n_repeat)
        crt.fit_importance(X, y)
        selected = crt.selection_fdr(fdr=fdr / 2, evalues=True, fdr_control="ebh")

        fdp, power = fdp_power(np.where(selected)[0], important_features)

        assert crt.test_scores_.shape == (n_repeat, n_features)
        assert fdp < 0.5
        assert power == 1.0

    def test_crt_invariant_with_bootstrap(self, data_generator):
        """Test repeat invariance"""
        fdr = 0.5
        X, y, important_features, _ = data_generator

        # Single AKO (or vanilla KO) (verbose vs no verbose)
        crt_repeat = CRT(
            n_repeat=5,
            generator=GaussianGenerator(
                cov_estimator=LedoitWolf(assume_centered=True), random_state=5
            ),
        )
        crt_repeat.fit_importance(X, y)
        selected_repeat = crt_repeat.selection_fdr(fdr=fdr)

        crt_no_repeat = CRT(
            n_repeat=1,
            generator=GaussianGenerator(
                cov_estimator=LedoitWolf(assume_centered=True), random_state=5
            ),
        )
        crt_no_repeat.fit(X).importance(X, y)
        selected_no_repeat = crt_no_repeat.selection_fdr(fdr=fdr)

        fdp_repeat, power_repeat = fdp_power(
            np.where(selected_repeat)[0], important_features
        )
        assert fdp_repeat < 0.5
        assert power_repeat == 1.0
        fdp_no_repeat, power_no_repeat = fdp_power(
            np.where(selected_no_repeat)[0], important_features
        )
        assert fdp_no_repeat < 0.5
        assert power_no_repeat == 1.0

        np.testing.assert_array_equal(
            crt_repeat.test_scores_[0],
            crt_no_repeat.test_scores_[0],
        )
        # test that the selection no boostract should be lower than with boostrap
        for i in np.where(selected_repeat)[0]:
            assert selected_no_repeat[i]
        # np.testing.assert_array_equal(
        # crt_bootstrap.importances_, crt_no_bootstrap.importances_
        # )
        # np.testing.assert_array_equal(crt_bootstrap.pvalues_, crt_no_bootstrap.pvalues_)
        # np.testing.assert_array_equal(selected_bootstrap, selected_no_bootstrap)

    def test_crt_function(self, data_generator):
        """Test the function crt"""
        X, y, important_features, _ = data_generator
        importance = crt(X, y, n_repeat=5)
        # check that importance scores are defined for each feature
        assert importance.shape == (X.shape[1],)
        # check that important features have the highest importance scores
        assert np.all(
            [int(i) in important_features for i in np.argsort(importance)[-5:]]
        )


##############################################################################
@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(300, 20, 5, 0.0, 42, 1.0, np.inf, 0.0)],
    ids=["default data"],
)
class TestCRTExceptions:
    """Test class for CRT exceptions"""

    def test_warning(self, data_generator):
        """Test if some warning are raised"""
        X, y, _, _ = data_generator
        crt = CRT(n_repeat=5)
        with pytest.warns(Warning, match="y won't be used"):
            crt.fit(X, y)
        with pytest.warns(Warning, match="cv won't be used"):
            crt.fit_importance(X, y, cv="test")

    def test_unfitted_importance(self, data_generator):
        """Test importance method with unfitted model"""
        X, y, _, _ = data_generator
        crt = CRT(n_repeat=5)

        with pytest.raises(
            ValueError,
            match="The CRT requires to be fitted before computing importance",
        ):
            crt.importance(X, y)

    def test_invalid_n_samplings(self, data_generator):
        """Test when invalid number of permutations is provided"""
        with pytest.raises(AssertionError, match="n_samplings must be positive"):
            CRT(n_repeat=-1)
