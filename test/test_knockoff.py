import numpy as np
import pytest
from sklearn.covariance import GraphicalLassoCV, LedoitWolf
from sklearn.linear_model import Lasso, LassoCV, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR
from sklearn.utils.estimator_checks import parametrize_with_checks

from hidimstat._utils.scenario import multivariate_simulation
from hidimstat.knockoffs import (
    ModelXKnockoff,
    model_x_knockoff_importance,
    set_alpha_max_lasso_path,
)
from hidimstat.samplers import GaussianKnockoffs
from hidimstat.statistical_tools.multiple_testing import fdp_power

from .conftest import SKLEARN_LT_1_6, check_estimator


def expected_failed_checks(estimator):
    if isinstance(estimator, (ModelXKnockoff, GaussianKnockoffs)):
        return {"check_fit2d_1sample": "TODO"}


ESTIMATORS_TO_CHECK = [ModelXKnockoff(), GaussianKnockoffs()]

if SKLEARN_LT_1_6:

    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(
            estimators=ESTIMATORS_TO_CHECK,
            return_expected_failed_checks=expected_failed_checks,
        ),
    )
    def test_check_estimator_sklearn_valid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)

    @pytest.mark.xfail(reason="invalid checks should fail")
    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(
            estimators=ESTIMATORS_TO_CHECK,
            valid=False,
            return_expected_failed_checks=expected_failed_checks,
        ),
    )
    def test_check_estimator_sklearn_invalid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)
else:

    @parametrize_with_checks(
        estimators=ESTIMATORS_TO_CHECK,
        expected_failed_checks=expected_failed_checks,
    )
    def test_check_estimator_sklearn(estimator, check):
        check(estimator)


def test_knockoff_bootstrap_quantile():
    """Test bootstrap knockoof with quantile aggregation"""
    n = 200
    p = 50
    signal_noise_ratio = 5
    n_repeats = 5
    fdr = 0.2
    fdp_list = []
    power_list = []
    for seed in range(10):
        X, y, beta, _ = multivariate_simulation(
            n, p, signal_noise_ratio=signal_noise_ratio, seed=seed
        )
        model_x_knockoff = ModelXKnockoff(
            estimator=LassoCV(),
            n_repeats=n_repeats,
            random_state=seed,
            n_jobs=5,
        ).fit(X, y)
        model_x_knockoff.importance()
        selected = model_x_knockoff.fdr_selection(fdr=fdr)

        fdp, power = fdp_power(selected, beta)
        fdp_list.append(fdp)
        power_list.append(power)

        assert model_x_knockoff.importances_.shape == (n_repeats, p)
        assert model_x_knockoff.pvalues_.shape == (n_repeats, p)
        assert model_x_knockoff.aggregated_pval_.shape == (p,)
    assert np.mean(fdp_list) < fdr
    assert np.mean(power_list) > 0.2


def test_knockoff_bootstrap_e_values():
    """Test bootstrap Knockoff with e-values"""
    n = 200
    p = 50
    signal_noise_ratio = 32
    n_repeats = 5
    fdr = 0.2
    fdp_list = []
    power_list = []
    for seed in range(10):
        X, y, beta, _ = multivariate_simulation(
            n, p, signal_noise_ratio=signal_noise_ratio, seed=seed
        )

        # Using e-values aggregation
        model_x_knockoff = ModelXKnockoff(
            estimator=LassoCV(),
            n_repeats=n_repeats,
            random_state=seed,
            n_jobs=5,
        ).fit(X, y)
        model_x_knockoff.importance()
        selected = model_x_knockoff.fdr_selection(
            fdr=fdr, fdr_control="ebh", evalues=True
        )

        fdp, power = fdp_power(selected, beta)
        fdp_list.append(fdp)
        power_list.append(power)

        assert model_x_knockoff.importances_.shape == (n_repeats, p)
        assert model_x_knockoff.pvalues_.shape == (n_repeats, p)
        assert model_x_knockoff.aggregated_eval_.shape == (p,)

    assert np.mean(fdp_list) < fdr
    assert np.mean(power_list) > 0.2


def test_invariant_with_bootstrap():
    """Test bootstrap Knockoff"""
    n = 200
    p = 50
    signal_noise_ratio = 5
    fdr = 0.2
    X, y, beta, _ = multivariate_simulation(
        n, p, signal_noise_ratio=signal_noise_ratio, seed=0
    )
    # Single AKO (or vanilla KO) (verbose vs no verbose)
    model_x_knockoff = ModelXKnockoff(
        estimator=LassoCV(),
        ko_generator=GaussianKnockoffs(
            cov_estimator=LedoitWolf(assume_centered=True)
        ),
        random_state=0,
        n_repeats=1,
    ).fit(X, y)
    model_x_knockoff.importance()
    selected = model_x_knockoff.fdr_selection(fdr=fdr)
    fdp_power(selected, beta)

    model_x_knockoff_repeat = ModelXKnockoff(
        estimator=LassoCV(),
        ko_generator=GaussianKnockoffs(
            cov_estimator=LedoitWolf(assume_centered=True)
        ),
        random_state=0,
        n_repeats=3,
    ).fit(X, y)
    model_x_knockoff_repeat.importance()
    selected_repeat = model_x_knockoff_repeat.fdr_selection(fdr=fdr)
    fdp_power(selected_repeat, beta)

    np.testing.assert_array_equal(
        model_x_knockoff.importances_[0],
        model_x_knockoff_repeat.importances_[0],
    )
    assert not np.array_equal(
        model_x_knockoff.pvalues_, model_x_knockoff_repeat.pvalues_
    )
    assert not np.array_equal(
        model_x_knockoff.importances_, model_x_knockoff_repeat.importances_
    )
    assert not np.array_equal(selected, selected_repeat)


def test_model_x_knockoff():
    """Test the selection of variable from knockoff"""
    fdr = 0.2
    n = 200
    p = 50
    support_size = 18
    fdp_list = []
    power_list = []
    for seed in range(10):
        X, y, beta, _ = multivariate_simulation(
            n, p, support_size=support_size, seed=seed
        )
        model_x_knockoff = ModelXKnockoff(
            estimator=LassoCV(), n_repeats=1, random_state=seed + 1
        ).fit(X, y)
        model_x_knockoff.importance()
        selected = model_x_knockoff.fdr_selection(fdr=fdr)

        fdp, power = fdp_power(selected, beta)
        fdp_list.append(fdp)
        power_list.append(power)

    assert np.mean(fdp_list) <= fdr
    assert np.mean(power_list) > 0.7


def test_model_x_knockoff_estimator():
    """Test knockoff with a crossvalidation estimator"""
    seed = 42
    fdr = 0.2
    n = 200
    p = 50
    X, y, beta, _ = multivariate_simulation(n, p, seed=seed)
    model_x_knockoff = ModelXKnockoff(
        n_repeats=1,
        estimator=GridSearchCV(
            Lasso(), param_grid={"alpha": np.linspace(0.2, 0.3, 5)}
        ),
        preconfigure_lasso_path=None,
    ).fit(X, y)
    model_x_knockoff.importance()
    selected = model_x_knockoff.fdr_selection(fdr=fdr)
    fdp_power(selected, beta)

    assert selected.shape == (p,)


def test_estimate_distribution():
    """
    Test different estimation of the covariance
    """
    seed = 42
    fdr = 0.1
    n = 100
    p = 50
    X, y, beta, _ = multivariate_simulation(n, p, seed=seed)
    generator = GaussianKnockoffs(
        cov_estimator=GraphicalLassoCV(
            alphas=[1e-3, 1e-2, 1e-1, 1],
            cv=KFold(n_splits=5),
        ),
    )
    model_x_knockoff = ModelXKnockoff(
        n_repeats=1, random_state=2, ko_generator=generator
    ).fit(X, y)
    model_x_knockoff.importance()
    selected = model_x_knockoff.fdr_selection(fdr=fdr)
    assert np.all(beta[selected])


def test_knockoff_function_not_centered():
    """Test function of knockoff not centered"""
    fdr = 0.2
    n = 100
    p = 50
    seed = 0
    X, y, beta, _ = multivariate_simulation(n, p, seed=seed)
    selected, importances, pvalues = model_x_knockoff_importance(
        X, y, centered=False, n_repeats=5, random_state=seed, fdr=fdr
    )
    fdp_power(selected, beta)
    assert selected.shape == (p,)
    assert importances.shape == (5, p)
    assert pvalues.shape == (5, p)


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
        model_x_knockoff = ModelXKnockoff(n_repeats=5)
        model_x_knockoff.fit(X, y)
        with pytest.warns(Warning, match="X won't be used"):
            model_x_knockoff.importance(X=X)
        with pytest.warns(Warning, match="y won't be used"):
            model_x_knockoff.importance(y=y)

    def test_error_lasso_statistic_with_sampling_with_bad_config(
        self, data_generator
    ):
        """Test error lasso statistic"""
        X, y, _, _ = data_generator
        model_x_knockoff = ModelXKnockoff(n_repeats=1, estimator=SVR())
        with pytest.raises(
            TypeError,
            match="You should not use this function to configure the estimator",
        ):
            model_x_knockoff.fit(X, y)

    def test_error_lasso_statistic_with_sampling(self, data_generator):
        """Test error lasso statistic"""
        X, y, _, _ = data_generator
        model_x_knockoff = ModelXKnockoff(
            n_repeats=1,
            estimator=SVR(),
            preconfigure_lasso_path=False,
        )
        with pytest.raises(TypeError, match="estimator should be linear"):
            model_x_knockoff.fit_importance(X, y)

    def test_unfitted_importance(self, data_generator):
        """Test importance method with unfitted model"""
        X, y, _, _ = data_generator
        model_x_knockoff = ModelXKnockoff(
            n_repeats=5,
            ko_generator=GaussianKnockoffs(
                cov_estimator=LedoitWolf(assume_centered=True)
            ),
            random_state=0,
        )

        with pytest.raises(
            ValueError,
            match="The Model-X Knockoff requires to be fitted before computing importance",
        ):
            model_x_knockoff.importance(X, y)

    def test_invalid_n_samplings(self, data_generator):  # noqa: ARG002
        """Test when invalid number of permutations is provided"""
        with pytest.raises(
            AssertionError, match="n_samplings must be positive"
        ):
            ModelXKnockoff(n_repeats=-1)


############################## test preconfigure #######################
def test_preconfigure_LassoCV(rng):
    """Test type errors"""
    with pytest.raises(
        TypeError,
        match="You should not use this function to configure the estimator",
    ):
        set_alpha_max_lasso_path(
            estimator=RidgeCV(),
            X=rng.random((10, 10)),
            y=rng.random(10),
            X_tilde=rng.random((10, 10)),
        )
