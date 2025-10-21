from hidimstat.knockoffs import model_x_knockoff
from hidimstat.knockoffs.data_simulation import simu_data
from hidimstat.knockoffs.utils import cal_fdp_power

seed = 0
fdr = 0.5


def test_model_x_knockoff():

    n = 300
    p = 100
    X, y, _, non_zero = simu_data(n, p, seed=seed)
    ko_result = model_x_knockoff(X, y, fdr=fdr, seed=seed + 1)
    fdp, power = cal_fdp_power(ko_result, non_zero)

    assert fdp <= 0.2
    assert power > 0.7
