import numpy as np
import pytest

from sklearn.covariance import LedoitWolf, GraphicalLassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold

from hidimstat import ModelXKnockoff
from hidimstat.knockoffs import model_x_knockoff
from hidimstat.statistical_tools.gaussian_distribution import GaussianDistribution
from hidimstat.statistical_tools.multiple_testing import fdp_power
from hidimstat.statistical_tools.lasso_test import lasso_statistic_with_sampling


def configure_linear_categorial_model_x_knockoff(X, y, n_repeat, seed, fdr):
    """
    Configure Model-X Knockoff model with linear regression
    for feature importance analysis.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix where each column represents a feature
        and each row a sample.
    y : array-like of shape (n_samples,)
        Target variable array.
    n_repeat : int
        Number of permutations to perform for the ModelXKnockoff analysis.
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
    1. Configures ModelXKnockoff
    2. Calculates feature importance
    The ModelXKnockoff method uses permutation-based importance scoring with linear
    regression as the base model.
    """
    # instantiate ModelXKnockoff model with linear regression imputer
    model_x_knockoff = ModelXKnockoff(
        n_repeat=n_repeat,
        generator=GaussianDistribution(
            cov_estimator=LedoitWolf(assume_centered=True), random_state=seed
        ),
    )
    # fit the model using the training set
    importance = model_x_knockoff.fit_importance(X, y)
    selected = model_x_knockoff.selection_fdr(fdr=fdr)
    return importance, selected


##############################################################################
## tests model_x_knockoff on different type of data
parameter_exact = [
    ("Dim", 500, 50, 10, 0.0, 42, 1.0, np.inf, 0.0),
    ("Dim with noise", 500, 50, 10, 0.0, 42, 1.0, 10.0, 0.0),
    ("Dim with correlated noise", 500, 50, 10, 0.0, 42, 1.0, 10.0, 0.2),
    ("Dim high level noise", 500, 50, 10, 0.2, 42, 1.0, 1.0, 0.0),
    ("Dim with correlated features and noise", 500, 50, 10, 0.2, 42, 1, 10, 0),
    (
        "Dim with correlated features and correlated noise",
        500,
        50,
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
    zip(*(list(zip(*parameter_exact))[1:])),
    ids=list(zip(*parameter_exact))[0],
)
@pytest.mark.parametrize(
    "model_x_knockoff_n_sampling, model_x_knockoff_seed, model_x_knockoff_fdr",
    [(10, 5, 0.4)],
    ids=["default_model_x_knockoff"],
)
def test_model_x_knockoff_linear_data_exact(
    data_generator,
    model_x_knockoff_n_sampling,
    model_x_knockoff_seed,
    model_x_knockoff_fdr,
):
    """Tests the method on linear cases with noise and correlation"""
    X, y, important_features, _ = data_generator
    importance, selected = configure_linear_categorial_model_x_knockoff(
        X, y, model_x_knockoff_n_sampling, model_x_knockoff_seed, model_x_knockoff_fdr
    )
    # check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)
    # check that important features have the highest importance scores
    assert np.all([int(i) in important_features for i in np.argsort(importance)[-10:]])

    # test the selection part
    fdp, power = fdp_power(np.where(selected)[0], important_features)
    assert fdp < model_x_knockoff_fdr
    assert power == 1.0


## tests model_x_knockoff on different type of data no power if correlated data
parameter_bad_FDR = [
    ("Dim with correlated features", 500, 50, 10, 0.2, 42, 1.0, np.inf, 0.0),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_bad_FDR))[1:])),
    ids=list(zip(*parameter_bad_FDR))[0],
)
@pytest.mark.parametrize(
    "model_x_knockoff_n_sampling, model_x_knockoff_seed, model_x_knockoff_fdr",
    [(10, 5, 0.4)],
    ids=["default_model_x_knockoff"],
)
def test_linear_data_bad_FDR(
    data_generator,
    model_x_knockoff_n_sampling,
    model_x_knockoff_seed,
    model_x_knockoff_fdr,
):
    """Tests the method on linear cases with noise and correlation"""
    X, y, important_features, _ = data_generator
    importance, selected = configure_linear_categorial_model_x_knockoff(
        X, y, model_x_knockoff_n_sampling, model_x_knockoff_seed, model_x_knockoff_fdr
    )
    # check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)
    # check that important features have the highest importance scores
    # assert np.all([int(i) in important_features for i in np.argsort(importance)[-10:]])

    # test the selection part
    fdp, power = fdp_power(np.where(selected)[0], important_features)
    assert fdp > model_x_knockoff_fdr
    assert power == 1.0


## tests model_x_knockoff no power
parameter_no_power = [
    ("Dim", 50, 150, 30, 0.0, 42, 1.0, np.inf, 0.0),
    ("Dim with noise", 50, 150, 30, 0.0, 42, 1.0, 10.0, 0.0),
    ("Dim with correlated noise", 50, 150, 30, 0.0, 42, 1.0, 10.0, 0.2),
    ("Dim with correlated features", 50, 150, 30, 0.2, 42, 1.0, np.inf, 0.0),
    ("Dim high level noise", 50, 150, 30, 0.2, 42, 1.0, 1.0, 0.0),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_no_power))[1:])),
    ids=list(zip(*parameter_no_power))[0],
)
@pytest.mark.parametrize(
    "model_x_knockoff_n_sampling, model_x_knockoff_seed, model_x_knockoff_fdr",
    [(10, 5, 0.1)],
    ids=["default_model_x_knockoff"],
)
def test_model_x_knockoff_linear_no_power(
    data_generator,
    model_x_knockoff_n_sampling,
    model_x_knockoff_seed,
    model_x_knockoff_fdr,
):
    """Tests the method on linear cases with noise and correlation"""
    X, y, important_features, _ = data_generator
    importance, selected = configure_linear_categorial_model_x_knockoff(
        X, y, model_x_knockoff_n_sampling, model_x_knockoff_seed, model_x_knockoff_fdr
    )
    # check that importance scores are defined for each feature
    assert importance.shape == (X.shape[1],)

    # test the selection part
    fdp, power = fdp_power(np.where(selected)[0], important_features)
    assert fdp < model_x_knockoff_fdr
    assert power == 0.0


## tests model_x_knockoff on different type of data
parameter_bad_detection = [
    ("No information", 500, 50, 10, 0.0, 42, 1.0, 0.0, 0.0),
    ("Dim with correlated features and noise", 50, 150, 30, 0.2, 42, 1, 10, 0),
    (
        "Dim with correlated features and correlated noise",
        50,
        150,
        30,
        0.2,
        42,
        1.0,
        10,
        0.2,
    ),
]


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    zip(*(list(zip(*parameter_bad_detection))[1:])),
    ids=list(zip(*parameter_bad_detection))[0],
)
@pytest.mark.parametrize(
    "model_x_knockoff_n_sampling, model_x_knockoff_seed, model_x_knockoff_fdr",
    [(10, 5, 0.4)],
    ids=["default_model_x_knockoff"],
)
def test_model_x_knockoff_linear_fail(
    data_generator,
    model_x_knockoff_n_sampling,
    model_x_knockoff_seed,
    model_x_knockoff_fdr,
):
    """Tests the method on linear cases with noise and correlation"""
    X, y, important_features, _ = data_generator
    importance, selected = configure_linear_categorial_model_x_knockoff(
        X, y, model_x_knockoff_n_sampling, model_x_knockoff_seed, model_x_knockoff_fdr
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
class TestModelXKnockoffClass:
    """Test the element of the class"""

    def test_model_x_knockoff_init(self, data_generator):
        """Test ModelXKnockoff initialization"""
        X, y, _, _ = data_generator
        model_x_knockoff = ModelXKnockoff()
        assert model_x_knockoff.n_jobs == 1
        assert model_x_knockoff.n_repeat == 1
        assert model_x_knockoff.generator is not None
        assert model_x_knockoff.statistical_test is not None

    def test_model_x_knockoff_fit(self, data_generator):
        """Test fitting ModelXKnockoff"""
        X, y, _, _ = data_generator
        model_x_knockoff = ModelXKnockoff()
        model_x_knockoff.fit(X)
        # check if the generator is fitted
        model_x_knockoff.generator._check_fit()

    def test_model_x_knockoff_importance(self, data_generator):
        """Test importance of ModelXKnockoff"""
        X, y, important_features, _ = data_generator
        model_x_knockoff = ModelXKnockoff(n_repeat=5)
        model_x_knockoff.fit(X)
        importance = model_x_knockoff.importance(X, y)

        # check that importance scores are defined for each feature
        assert importance.shape == (X.shape[1],)
        # check that important features have the highest importance scores
        assert np.all(
            [int(i) in important_features for i in np.argsort(importance)[-5:]]
        )

    def test_model_x_knockoff_categorical(
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
        """Test ModelXKnockoff with categorical variables"""
        rng = np.random.default_rng(seed)
        X_cont = rng.random((n_samples, 2))
        X_cat = rng.integers(low=0, high=3, size=(n_samples, 1))
        X = np.hstack([X_cont, X_cat])
        y = rng.random((n_samples))

        model_x_knockoff = ModelXKnockoff()

        with pytest.warns(
            Warning,
            match="The conditional covariance matrix for knockoffs is not positive definite.",
        ):
            with pytest.warns(
                Warning,
                match="The equi-correlated matrix for knockoffs is not positive definite.",
            ):
                importances = model_x_knockoff.fit_importance(X, y)
        assert len(importances) == 3
        assert np.all(importances >= 0)

    def test_model_x_knockoff_CV_estimator(self, data_generator, seed):
        """Test ModelXKnockoff with a crossvalidation estimator"""
        fdr = 0.7
        X, y, important_features, _ = data_generator

        def lasso_statistic_gen(X, X_tilde, y):
            return lasso_statistic_with_sampling(
                X,
                X_tilde,
                y,
                lasso=GridSearchCV(
                    Lasso(), param_grid={"alpha": np.linspace(0.2, 0.3, 5)}
                ),
                preconfigure_lasso=None,
            )

        model_x_knockoff = ModelXKnockoff(
            n_repeat=1,
            generator=GaussianDistribution(
                cov_estimator=LedoitWolf(assume_centered=True), random_state=seed + 2
            ),
            statistical_test=lasso_statistic_gen,
        )
        model_x_knockoff.fit_importance(X, y)
        selected = model_x_knockoff.selection_fdr(fdr=fdr)

        fdp, power = fdp_power(np.where(selected)[0], important_features)

        assert fdp <= fdr
        assert power == 1.0

    def test_estimate_distribution(self, data_generator, seed):
        """Test different estimation of the covariance"""
        fdr = 0.3
        X, y, important_features, _ = data_generator
        model_x_knockoff = ModelXKnockoff(
            n_repeat=1,
            generator=GaussianDistribution(
                cov_estimator=LedoitWolf(assume_centered=True), random_state=seed + 1
            ),
        )
        model_x_knockoff.fit_importance(X, y)
        selected = model_x_knockoff.selection_fdr(fdr=fdr)
        for i in important_features:
            assert selected[i]

        model_x_knockoff = ModelXKnockoff(
            n_repeat=1,
            generator=GaussianDistribution(
                cov_estimator=GraphicalLassoCV(
                    alphas=[1e-3, 1e-2, 1e-1, 1],
                    cv=KFold(n_splits=5, shuffle=True, random_state=seed + 2),
                ),
                random_state=seed + 3,
            ),
        )
        model_x_knockoff.fit_importance(X, y)
        selected = model_x_knockoff.selection_fdr(fdr=fdr)
        for i in important_features:
            assert selected[i]

    def test_model_x_knockoff_selection(self, data_generator):
        """Test the selection of variable from knockoff"""
        fdr = 0.5
        n_repeat = 5
        X, y, important_features, _ = data_generator
        model_x_knockoff = ModelXKnockoff(n_repeat=n_repeat)
        model_x_knockoff.fit_importance(X, y)
        selected = model_x_knockoff.selection_fdr(fdr=fdr)
        print(model_x_knockoff.importances_, model_x_knockoff.pvalues_)
        fdp, power = fdp_power(np.where(selected)[0], important_features)
        assert fdp <= 0.5
        assert power > 0.7
        assert np.all(0 <= model_x_knockoff.pvalues_) or np.all(
            model_x_knockoff.pvalues_ <= 1
        )

    def test_model_x_knockoff_repeat_quantile(self, data_generator, n_features):
        """Test ModelXKnockoff selection"""
        fdr = 0.5
        n_repeat = 5
        X, y, important_features, _ = data_generator
        model_x_knockoff = ModelXKnockoff(n_repeat=n_repeat)
        model_x_knockoff.fit_importance(X, y)
        selected = model_x_knockoff.selection_fdr(fdr=fdr)

        fdp, power = fdp_power(np.where(selected)[0], important_features)

        assert model_x_knockoff.test_scores_.shape == (n_repeat, n_features)
        assert fdp < 0.5
        assert power == 1.0

    def test_model_x_knockoff_repeat_e_values(self, data_generator, n_features):
        """Test ModelXKnockoff selection with e-values"""
        fdr = 0.5
        n_repeat = 5
        X, y, important_features, _ = data_generator
        model_x_knockoff = ModelXKnockoff(
            n_repeat=n_repeat,
            generator=GaussianDistribution(
                cov_estimator=LedoitWolf(assume_centered=True), random_state=0
            ),
        )
        model_x_knockoff.fit_importance(X, y)
        selected = model_x_knockoff.selection_fdr(
            fdr=fdr, evalues=True, fdr_control="ebh"
        )

        fdp, power = fdp_power(np.where(selected)[0], important_features)

        assert model_x_knockoff.test_scores_.shape == (n_repeat, n_features)
        assert fdp < 0.5
        assert power == 1.0

    def test_model_x_knockoff_invariant_with_bootstrap(self, data_generator):
        """Test repeat invariance"""
        fdr = 0.5
        X, y, important_features, _ = data_generator

        # Single AKO (or vanilla KO) (verbose vs no verbose)
        model_x_knockoff_repeat = ModelXKnockoff(
            n_repeat=10,
            generator=GaussianDistribution(
                cov_estimator=LedoitWolf(assume_centered=True), random_state=0
            ),
        )
        model_x_knockoff_repeat.fit_importance(X, y)
        selected_repeat = model_x_knockoff_repeat.selection_fdr(fdr=fdr)

        model_x_knockoff_no_repeat = ModelXKnockoff(
            n_repeat=1,
            generator=GaussianDistribution(
                cov_estimator=LedoitWolf(assume_centered=True), random_state=0
            ),
        )
        model_x_knockoff_no_repeat.fit(X).importance(X, y)
        selected_no_repeat = model_x_knockoff_no_repeat.selection_fdr(fdr=fdr)

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
            model_x_knockoff_repeat.test_scores_[0],
            model_x_knockoff_no_repeat.test_scores_[0],
        )
        # test that the selection no boostract should be lower than with boostrap
        for i in np.where(selected_repeat)[0]:
            assert selected_no_repeat[i]
        # np.testing.assert_array_equal(
        #     model_x_knockoff_repeat.importances_,
        #     model_x_knockoff_no_repeat.importances_,
        # )
        # np.testing.assert_array_equal(
        #     model_x_knockoff_repeat.pvalues_, model_x_knockoff_no_repeat.pvalues_
        # )
        # np.testing.assert_array_equal(selected_repeat, selected_no_repeat)

    def test_model_x_knockoff_function(self, data_generator):
        """Test the function ModelXKnockoff"""
        X, y, important_features, _ = data_generator
        importance = model_x_knockoff(X, y, n_repeat=5)
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
class TestModelXKnockoffExceptions:
    """Test class for ModelXKnockoff exceptions"""

    def test_warning(self, data_generator):
        """Test if some warning are raised"""
        X, y, _, _ = data_generator
        model_x_knockoff = ModelXKnockoff(n_repeat=5)
        with pytest.warns(Warning, match="y won't be used"):
            model_x_knockoff.fit(X, y)
        with pytest.warns(Warning, match="cv won't be used"):
            model_x_knockoff.fit_importance(X, y, cv="test")

    def test_unfitted_importance(self, data_generator):
        """Test importance method with unfitted model"""
        X, y, _, _ = data_generator
        model_x_knockoff = ModelXKnockoff(n_repeat=5)

        with pytest.raises(
            ValueError,
            match="The Model-X Knockoff requires to be fitted before computing importance",
        ):
            model_x_knockoff.importance(X, y)

    def test_invalid_n_samplings(self, data_generator):
        """Test when invalid number of permutations is provided"""
        with pytest.raises(AssertionError, match="n_samplings must be positive"):
            ModelXKnockoff(n_repeat=-1)
