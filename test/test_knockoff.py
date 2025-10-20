import numpy as np
import pytest
from sklearn.covariance import GraphicalLassoCV, LedoitWolf
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold

from hidimstat._utils.scenario import multivariate_simulation
from hidimstat.knockoffs import ModelXKnockoff, model_x_knockoff
from hidimstat.statistical_tools.gaussian_knockoffs import GaussianKnockoffs
from hidimstat.statistical_tools.lasso_test import lasso_statistic_with_sampling
from hidimstat.statistical_tools.multiple_testing import fdp_power


def test_knockoff_bootstrap_quantile():
    """Test bootstrap knockoof with quantile aggregation"""
    n = 500
    p = 100
    signal_noise_ratio = 5
    n_repeat = 25
    fdr = 0.5
    X, y, beta, noise = multivariate_simulation(
        n, p, signal_noise_ratio=signal_noise_ratio, seed=0
    )
    modelxknockoff = ModelXKnockoff(n_repeat=n_repeat).fit(X)
    modelxknockoff.importance(X, y)
    selected = modelxknockoff.fdr_selection(fdr=fdr)

    fdp, power = fdp_power(np.where(selected)[0], np.where(beta)[0])

    assert modelxknockoff.test_scores_.shape == (n_repeat, p)
    assert modelxknockoff.importances_.shape == (p,)
    assert modelxknockoff.pvalues_.shape == (p,)
    assert modelxknockoff.aggregated_pval_.shape == (p,)
    assert modelxknockoff.pvalues_.shape == (p,)
    assert fdp < 0.5
    assert power > 0.1


def test_knockoff_bootstrap_e_values():
    """Test bootstrap Knockoff with e-values"""
    n = 500
    p = 100
    signal_noise_ratio = 5
    n_repeat = 25
    fdr = 0.5
    X, y, beta, noise = multivariate_simulation(
        n, p, signal_noise_ratio=signal_noise_ratio, seed=0
    )

    # Using e-values aggregation
    modelxknockoff = ModelXKnockoff(n_repeat=n_repeat).fit(X)
    modelxknockoff.importance(X, y)
    selected = modelxknockoff.fdr_selection(
        fdr=fdr / 2, fdr_control="ebh", evalues=True
    )

    fdp, power = fdp_power(np.where(selected)[0], np.where(beta)[0])

    assert modelxknockoff.test_scores_.shape == (n_repeat, p)
    assert modelxknockoff.importances_.shape == (p,)
    assert modelxknockoff.pvalues_.shape == (p,)
    assert modelxknockoff.aggregated_eval_.shape == (p,)
    assert modelxknockoff.pvalues_.shape == (p,)
    assert fdp < 0.5
    assert power > 0.1


def test_invariant_with_bootstrap():
    """Test bootstrap Knockoff"""
    n = 500
    p = 100
    signal_noise_ratio = 5
    fdr = 0.8
    X, y, beta, noise = multivariate_simulation(
        n, p, signal_noise_ratio=signal_noise_ratio, seed=0
    )
    # Single AKO (or vanilla KO) (verbose vs no verbose)
    modelxknockoff = ModelXKnockoff(
        generator=GaussianKnockoffs(cov_estimator=LedoitWolf(assume_centered=True)),
        random_state=0,
        n_repeat=1,
    ).fit(X)
    modelxknockoff.importance(X, y)
    selected = modelxknockoff.fdr_selection(fdr=fdr)
    fdp, power = fdp_power(np.where(selected)[0], np.where(beta)[0])

    modelxknockoff_repeat = ModelXKnockoff(
        generator=GaussianKnockoffs(cov_estimator=LedoitWolf(assume_centered=True)),
        random_state=0,
        n_repeat=5,
    ).fit(X)
    modelxknockoff_repeat.importance(X, y)
    selected_repeat = modelxknockoff_repeat.fdr_selection(fdr=fdr)
    fdp_repeat, power_repeat = fdp_power(np.where(selected)[0], np.where(beta)[0])

    np.testing.assert_array_equal(
        modelxknockoff.test_scores_[0], modelxknockoff_repeat.test_scores_[0]
    )
    assert not np.array_equal(modelxknockoff.pvalues_, modelxknockoff_repeat.pvalues_)
    assert not np.array_equal(
        modelxknockoff.importances_, modelxknockoff_repeat.importances_
    )
    assert fdp_repeat <= fdp
    assert power_repeat <= power
    assert not np.array_equal(selected, selected_repeat)


def test_model_x_knockoff():
    """Test the selection of variable from knockoff"""
    seed = 42
    fdr = 0.2
    n = 300
    p = 300
    support_size = 18
    X, y, beta, noise = multivariate_simulation(
        n, p, support_size=support_size, seed=seed
    )
    modelxknockoff = ModelXKnockoff(n_repeat=1).fit(X)
    modelxknockoff.importance(X, y)
    selected = modelxknockoff.fdr_selection(fdr=fdr)

    fdp, power = fdp_power(np.where(selected)[0], np.where(beta)[0])
    assert fdp <= 0.2
    assert power > 0.7
    assert np.all(0 <= modelxknockoff.pvalues_) or np.all(modelxknockoff.pvalues_ <= 1)


def test_model_x_knockoff_estimator():
    """Test knockoff with a crossvalidation estimator"""

    def statistical_test(
        X,
        X_tilde,
        y,
    ):
        return lasso_statistic_with_sampling(
            X,
            X_tilde,
            y,
            lasso=GridSearchCV(Lasso(), param_grid={"alpha": np.linspace(0.2, 0.3, 5)}),
            preconfigure_lasso=None,
        )

    seed = 42
    fdr = 0.2
    n = 300
    p = 300
    X, y, beta, noise = multivariate_simulation(n, p, seed=seed)
    modelxknockoff = ModelXKnockoff(n_repeat=1, statistical_test=statistical_test).fit(
        X
    )
    modelxknockoff.importance(X, y)
    selected = modelxknockoff.fdr_selection(fdr=fdr)
    fdp, power = fdp_power(np.where(selected)[0], np.where(beta)[0])

    assert fdp <= fdr
    assert power > 0.7


def test_estimate_distribution():
    """
    test different estimation of the covariance
    """
    seed = 42
    fdr = 0.1
    n = 100
    p = 50
    X, y, beta, noise = multivariate_simulation(n, p, seed=seed)
    generator = GaussianKnockoffs(
        cov_estimator=GraphicalLassoCV(
            alphas=[1e-3, 1e-2, 1e-1, 1],
            cv=KFold(n_splits=5),
        ),
    )
    modelxknockoff = ModelXKnockoff(
        n_repeat=1, random_state=2, generator=generator
    ).fit(X)
    modelxknockoff.importance(X, y)
    selected = modelxknockoff.fdr_selection(fdr=fdr)
    assert np.all(beta[selected])


def test_knockoff_function_not_centered():
    """Test function of knockoff not centered"""
    seed = 42
    fdr = 0.1
    n = 100
    p = 50
    X, y, beta, noise = multivariate_simulation(n, p, seed=seed)
    selected, importances, pvalues = model_x_knockoff(X, y, centered=False)
    fdp, power = fdp_power(np.where(selected)[0], np.where(beta)[0])
    assert selected.shape == (p,)
    assert importances.shape == (p,)
    assert pvalues.shape == (p,)
    assert fdp <= 0.2
    assert power > 0.8


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
        model_x_knockoff = ModelXKnockoff(
            n_repeat=5,
            generator=GaussianKnockoffs(cov_estimator=LedoitWolf(assume_centered=True)),
            random_state=0,
        )

        with pytest.raises(
            ValueError,
            match="The Model-X Knockoff requires to be fitted before computing importance",
        ):
            model_x_knockoff.importance(X, y)

    def test_invalid_n_samplings(self, data_generator):
        """Test when invalid number of permutations is provided"""
        with pytest.raises(AssertionError, match="n_samplings must be positive"):
            ModelXKnockoff(n_repeat=-1)
