from hidimstat.knockoffs import model_x_knockoff_aggregation, model_x_knockoff,\
    model_x_knockoff_filter, model_x_knockoff_pvalue, model_x_knockoff_bootstrap_e_value,\
    model_x_knockoff_bootstrap_quantile
from hidimstat.gaussian_knockoff import gaussian_knockoff_generation
from hidimstat.data_simulation import simu_data
from hidimstat.utils import cal_fdp_power
import numpy as np
import pytest
from sklearn.covariance import LedoitWolf, GraphicalLassoCV




def test_knockoff_aggregation():
    n = 500
    p = 100
    snr = 5
    n_bootstraps = 25
    fdr = 0.5
    X, y, _, non_zero_index = simu_data(n, p, snr=snr, seed=0)
    
    test_scores = model_x_knockoff_aggregation(
        X, y, n_bootstraps=n_bootstraps, random_state=0
    )
    selected_verbose, aggregated_pval, pvals = model_x_knockoff_bootstrap_quantile(test_scores, fdr=fdr, selection_only=False)
    
    fdp_verbose, power_verbose = cal_fdp_power(selected_verbose, non_zero_index)

    selected_no_verbose = model_x_knockoff_bootstrap_quantile(test_scores, fdr=fdr, selection_only=True)

    fdp_no_verbose, power_no_verbose = cal_fdp_power(
        selected_no_verbose, non_zero_index
    )

    assert pvals.shape == (n_bootstraps, p)
    assert fdp_verbose < 0.5
    assert power_verbose > 0.1
    assert fdp_no_verbose < 0.5
    assert power_no_verbose > 0.1
    np.testing.assert_array_equal(selected_no_verbose, selected_verbose)

    # Single AKO (or vanilla KO) (verbose vs no verbose)
    test_bootstrap = model_x_knockoff_aggregation(
        X, y, n_bootstraps=2, random_state=5
    )
    test_no_bootstrap = model_x_knockoff(X, y, seed=np.random.RandomState(5).randint(1, np.iinfo(np.int32).max, 2)[0])

    np.testing.assert_array_equal(test_bootstrap[0], test_no_bootstrap)

    # Using e-values aggregation (verbose vs no verbose)

    selected_verbose, aggregated_eval, evals = model_x_knockoff_bootstrap_e_value(test_scores,  fdr=fdr, selection_only=False)
    fdp_verbose, power_verbose = cal_fdp_power(selected_verbose, non_zero_index)
    selected_no_verbose = model_x_knockoff_bootstrap_e_value(test_scores,  fdr=fdr, selection_only=True)
    fdp_no_verbose, power_no_verbose = cal_fdp_power(
        selected_no_verbose, non_zero_index
    )

    assert pvals.shape == (n_bootstraps, p)
    assert fdp_verbose < 0.5
    assert power_verbose > 0.1
    assert fdp_no_verbose < 0.5
    assert power_no_verbose > 0.1

    # Checking wrong type for random_state
    with pytest.raises(Exception):
        _ = model_x_knockoff_aggregation(
            X,
            y,
            random_state="test",
        )

    # Checking value for offset not belonging to (0,1)
    with pytest.raises(Exception):
        _ = model_x_knockoff_bootstrap_quantile(
            test_scores,
            offset=2,
        )

    with pytest.raises(Exception):
        _ = model_x_knockoff_bootstrap_e_value(
            test_scores,
            offset=2,
        )

def test_model_x_knockoff():
    
    seed = 42
    fdr = 0.2
    n = 300
    p = 300
    X, y, _, non_zero = simu_data(n, p, seed=seed)
    test_score = model_x_knockoff(X, y, seed=seed + 1)
    ko_result = model_x_knockoff_filter(test_score, fdr=fdr,)
    fdp, power = cal_fdp_power(ko_result, non_zero)

    assert fdp <= 0.2
    assert power > 0.7


def test_estimate_distribution():
    """
    TODO replace the estimate distribution by calling knockoff function with them
    """
    SEED = 42
    fdr = 0.1
    n = 100
    p = 50
    X, y, _, non_zero = simu_data(n, p, seed=SEED)
    mu = X.mean(axis=0)
    Sigma = LedoitWolf(assume_centered=True).fit(X).covariance_
    

    assert mu.size == p
    assert Sigma.shape == (p, p)

    mu = X.mean(axis=0)
    Sigma = GraphicalLassoCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X).covariance_

    assert mu.size == p
    assert Sigma.shape == (p, p)


def test_gaussian_knockoff_equi():
    SEED = 42
    fdr = 0.1
    n = 100
    p = 50
    X, y, _, non_zero = simu_data(n, p, seed=SEED)
    mu = X.mean(axis=0)
    Sigma = LedoitWolf(assume_centered=True).fit(X).covariance_

    X_tilde = gaussian_knockoff_generation(X, mu, Sigma, seed=SEED * 2)

    assert X_tilde.shape == (n, p)

