"""
Knockoff aggregation on simulated data
=============================

In this example, we show an example of variable selection using
model-X Knockoffs introduced by Candès et al. in [1]. A notable
drawback of this procedure is the randomness associated with 
the knockoff generation process. This can result in unstable 
infernce.

This example exhibits the two aggregation procedures described 
by Nguyen et al. [2] and Ren et al. [3] to derandomize inference.

References
----------
.. [1] Candes, Emmanuel, et al. "Panning for gold:'model-X' knockoffs
       for high dimensional controlled variable selection." 
       Journal of the Royal Statistical Society: Series B (Statistical Methodology)
       80.3 (2018): 551-577.

.. [2] Nguyen, Tuan-Binh, et al. "Aggregation of multiple knockoffs." 
       International Conference on Machine Learning. PMLR, 2020.

.. [3] Ren, Zhimei, and Rina Foygel Barber. 
       "Derandomized knockoffs: leveraging e-values for 
       false discovery rate control."
       arXiv preprint arXiv:2205.15461 (2022).

"""

#############################################################################
# Imports needed for this script
# ------------------------------

import numpy as np
from hidimstat.data_simulation import simu_data
from hidimstat.knockoffs import model_x_knockoff
from hidimstat.knockoff_aggregation import knockoff_aggregation
from hidimstat.utils import cal_fdp_power
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt

n_subjects = 500
n_clusters = 500
rho = 0.7
sparsity = 0.1
fdr = 0.1
seed = 45
n_bootstraps = 25
n_jobs = 25
runs = 20

rng = check_random_state(seed)
seed_list = rng.randint(1, np.iinfo(np.int32).max, runs)


def single_run(
    n_subjects, n_clusters, rho, sparsity, fdr, n_bootstraps, n_jobs, seed=None
):
    # Generate data
    X, y, _, non_zero_index = simu_data(
        n_subjects, n_clusters, rho=rho, sparsity=sparsity, seed=seed
    )

    # Use p-values aggregation [2]
    aggregated_ko_selection = knockoff_aggregation(
        X,
        y,
        fdr=fdr,
        n_bootstraps=n_bootstraps,
        n_jobs=n_jobs,
        gamma=0.3,
        random_state=seed,
    )

    fdp_pval, power_pval = cal_fdp_power(aggregated_ko_selection, non_zero_index)

    # Use e-values aggregation [1]
    eval_selection = knockoff_aggregation(
        X,
        y,
        fdr=fdr,
        method="e-values",
        n_bootstraps=n_bootstraps,
        n_jobs=n_jobs,
        gamma=0.3,
        random_state=seed,
    )

    fdp_eval, power_eval = cal_fdp_power(eval_selection, non_zero_index)

    return fdp_pval, fdp_eval, power_pval, power_eval


fdps_pval = []
fdps_eval = []
powers_pval = []
powers_eval = []

for seed in seed_list:
    fdp_mx, fdp_agg, power_mx, power_agg = single_run(
        n_subjects, n_clusters, rho, sparsity, fdr, n_bootstraps, n_jobs, seed=seed
    )
    fdps_pval.append(fdp_mx)
    fdps_eval.append(fdp_agg)

    powers_pval.append(power_mx)
    powers_eval.append(power_agg)

# Plot FDP and Power distributions

fdps = [fdps_pval, fdps_eval]
powers = [powers_pval, powers_eval]


def plot_results(bounds, fdr, nsubjects, n_clusters, rho, power=False):
    plt.figure()
    for nb in range(len(bounds)):
        for i in range(len(bounds[nb])):
            y = bounds[nb][i]
            x = np.random.normal(nb + 1, 0.05)
            plt.scatter(x, y, alpha=0.65, c="blue")

    plt.boxplot(bounds, sym="")
    if power:
        plt.xticks([1, 2], ["Quantile aggregation", "e-values aggregation"])
        plt.title(f"FDR = {fdr}, n = {nsubjects}, p = {n_clusters}, rho = {rho}")
        plt.ylabel("Empirical Power")

    else:
        plt.hlines(fdr, xmin=0.8, xmax=2.3, label="Requested FDR control", color="red")
        plt.xticks([1, 2], ["Quantile aggregation", "e-values aggregation"])
        plt.title(f"FDR = {fdr}, n = {nsubjects}, p = {n_clusters}, rho = {rho}")
        plt.ylabel("Empirical FDP")
        plt.legend(loc="best")

    plt.show()


plot_results(fdps, fdr, n_subjects, n_clusters, rho)
plot_results(powers, fdr, n_subjects, n_clusters, rho, power=True)
