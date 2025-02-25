from hidimstat import model_x_knockoff
from hidimstat.data_simulation import simu_data
from hidimstat.utils import cal_fdp_power


def test_model_x_knockoff():
    seed = 42
    fdr = 0.2
    n = 300
    p = 300
    X, y, _, non_zero = simu_data(n, p, seed=seed)
    ko_result = model_x_knockoff(X, y, fdr=fdr, seed=seed + 1)
    fdp, power = cal_fdp_power(ko_result, non_zero)

    assert fdp <= 0.2
    assert power > 0.7


def test_model_x_knockoff_with_verbose():
    seed = 42
    fdr = 0.2
    n = 300
    p = 300
    X, y, _, non_zero = simu_data(n, p, seed=seed)
    ko_result, test_scored, thres, X_tilde = model_x_knockoff(
        X, y, fdr=fdr, seed=5, verbose=True
    )
    # TODO add tests for the 3 other variables

    fdp, power = cal_fdp_power(ko_result, non_zero)

    assert fdp <= 0.2
    assert power > 0.7
