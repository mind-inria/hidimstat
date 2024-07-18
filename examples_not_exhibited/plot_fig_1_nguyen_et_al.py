# Authors: Binh Nguyen <tuan-binh.nguyen@inria.fr>
"""
Work in Progress : Histogram of KO vs AKO performance
=====================================================

Example: reproducing Figure 1 in::

    Nguyen et al. (2020) Aggregation of Multiple Knockoffs
    https://arxiv.org/abs/2002.09269

To reduce the script runtime it is desirable to increase n_jobs parameter.
"""
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

from hidimstat.knockoffs import knockoff_aggregation, model_x_knockoff
from hidimstat.knockoffs.data_simulation import simu_data
from hidimstat.knockoffs.utils import cal_fdp_power

color_blue = "#1f77b4"
color_teal = "#1fbecf"


def one_inference(
    n,
    p,
    snr,
    rho,
    sparsity,
    n_bootstraps=25,
    gamma=0.3,
    n_jobs=1,
    offset=1,
    fdr=0.1,
    seed=None,
):

    # Simulate data following autoregressive structure, seed is fixed to ensure
    # doing inference on only 1 simulation
    X, y, _, non_zero_index = simu_data(
        n=n, p=p, rho=rho, snr=snr, sparsity=sparsity, seed=42
    )
    X = StandardScaler().fit_transform(X)

    # Single knockoff -- has to do it 25 times to match the number of
    # bootstraps in AKO for fair comparison
    ko_fdps = []
    ko_powers = []

    for i in range(n_bootstraps):
        ko_selected = model_x_knockoff(
            X, y, fdr=fdr, offset=offset, n_jobs=n_jobs, seed=n_bootstraps * seed
        )
        ko_fdp, ko_power = cal_fdp_power(ko_selected, non_zero_index)
        ko_fdps.append(ko_fdp)
        ko_powers.append(ko_power)

    # Aggregated knockoff
    ako_selected = knockoff_aggregation(
        X,
        y,
        fdr=fdr,
        offset=offset,
        n_jobs=n_jobs,
        gamma=gamma,
        n_bootstraps=n_bootstraps,
        random_state=seed * 2,
    )

    ako_fdp, ako_power = cal_fdp_power(ako_selected, non_zero_index)

    return ko_fdps, ako_fdp, ko_powers, ako_power


def plot(results, n_simu, fdr):

    ko_fdps = np.array([results[i][0] for i in range(n_simu)]).ravel()
    ako_fdps = np.array([results[i][1] for i in range(n_simu)]).ravel()
    ko_powers = np.array([results[i][2] for i in range(n_simu)]).ravel()
    ako_powers = np.array([results[i][3] for i in range(n_simu)]).ravel()

    # Plotting
    n_bins = 30
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 4))
    ax1.tick_params(labelsize=14)
    ax1.hist(ko_fdps, edgecolor="k", range=[0.0, 1.0], bins=n_bins, color=color_blue)
    ax1.axvline(x=fdr, linestyle="--", color="r", linewidth=1.0)
    ax2.tick_params(labelsize=14)
    ax2.hist(ko_powers, edgecolor="k", range=[0.0, 1.0], bins=n_bins, color=color_blue)
    ax3.tick_params(labelsize=14)
    ax3.hist(ako_fdps, edgecolor="k", range=[0.0, 1.0], bins=n_bins, color=color_teal)
    ax3.axvline(x=fdr, linestyle="--", color="r", linewidth=1.0)
    ax4.tick_params(labelsize=14)
    ax4.hist(ako_powers, edgecolor="k", range=[0.0, 1.0], bins=n_bins, color=color_teal)
    plt.tight_layout()

    figname = "figures/histogram_ko_vs_ako.png"
    plt.savefig(figname)
    print(f"Save figure to {figname}")


def main():
    # Simulation paramaters
    n, p = 50, 200
    snr = 3.0
    rho = 0.5
    sparsity = 0.06
    offset = 1
    fdr = 0.05
    gamma = 0.3
    n_bootstraps = 10
    n_simu = 10
    offset = 1

    results = Parallel(n_jobs=1)(
        delayed(one_inference)(
            n=n,
            p=p,
            snr=snr,
            rho=rho,
            sparsity=sparsity,
            n_jobs=1,
            n_bootstraps=n_bootstraps,
            fdr=fdr,
            offset=offset,
            gamma=gamma,
            seed=seed,
        )
        for seed in range(n_simu)
    )

    # Plotting
    plot(results, n_simu, fdr)
    print("Done!")


# if __name__ == '__main__':
#     main()
