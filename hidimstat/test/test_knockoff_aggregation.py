from hidimstat import knockoff_aggregation, model_x_knockoff
from hidimstat.data_simulation import simu_data
from hidimstat.utils import cal_fdp_power
import numpy as np
import pytest

n = 500
p = 100
snr = 5
n_bootstraps = 25
fdr = 0.5
X, y, _, non_zero_index = simu_data(n, p, snr=snr, seed=0)


def test_knockoff_aggregation():
    selected_verbose, aggregated_pval, pvals = knockoff_aggregation(
        X, y, fdr=fdr, n_bootstraps=n_bootstraps, verbose=True, random_state=0
    )

    fdp_verbose, power_verbose = cal_fdp_power(selected_verbose, non_zero_index)

    selected_no_verbose = knockoff_aggregation(
        X, y, fdr=fdr, n_bootstraps=n_bootstraps, verbose=False, random_state=None
    )

    fdp_no_verbose, power_no_verbose = cal_fdp_power(
        selected_no_verbose, non_zero_index
    )

    assert pvals.shape == (n_bootstraps, p)
    assert fdp_verbose < 0.5
    assert power_verbose > 0.1
    assert fdp_no_verbose < 0.5
    assert power_no_verbose > 0.1

    # Single AKO (or vanilla KO) (verbose vs no verbose)
    selected_no_verbose = knockoff_aggregation(
        X, y, fdr=fdr, verbose=False, n_bootstraps=1, random_state=5
    )

    selected_verbose, pvals = knockoff_aggregation(
        X, y, fdr=fdr, verbose=True, n_bootstraps=1, random_state=5
    )

    selected_ko = model_x_knockoff(X, y, fdr=fdr, seed=5)

    np.testing.assert_array_equal(selected_no_verbose, selected_ko)

    fdp_no_verbose, power_no_verbose = cal_fdp_power(
        selected_no_verbose, non_zero_index
    )
    fdp_verbose, power_verbose = cal_fdp_power(selected_verbose, non_zero_index)

    assert fdp_verbose < 0.5
    assert power_verbose > 0.1

    # Using e-values aggregation (verbose vs no verbose)

    selected_verbose, aggregated_pval, pvals = knockoff_aggregation(
        X,
        y,
        fdr=fdr,
        n_bootstraps=n_bootstraps,
        method="e-values",
        verbose=True,
        random_state=0,
    )

    fdp_verbose, power_verbose = cal_fdp_power(selected_verbose, non_zero_index)

    selected_no_verbose = knockoff_aggregation(
        X,
        y,
        fdr=fdr,
        n_bootstraps=n_bootstraps,
        method="e-values",
        verbose=False,
        random_state=None,
    )

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
        _ = knockoff_aggregation(
            X,
            y,
            random_state="test",
        )

    # Checking value for offset not belonging to (0,1)
    with pytest.raises(Exception):
        _ = knockoff_aggregation(
            X,
            y,
            offset=2,
            method="quantile",
            random_state="test",
        )

    with pytest.raises(Exception):
        _ = knockoff_aggregation(
            X,
            y,
            offset=2,
            method="e-values",
            random_state="test",
        )
