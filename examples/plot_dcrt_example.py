"""
This example compares the performance of d0crt based on
the lasso (1) and random forest (2) implementations. The number of
repetitions is set to 100. The metrics used are the type-I error and
the power
"""

import numpy as np
import matplotlib.pyplot as plt
from hidimstat.dcrt import dcrt_zero
from hidimstat.scenario import multivariate_1D_simulation

typeI_error = {"Lasso": [], "Forest": []}
power = {"Lasso": [], "Forest": []}

for sim_ind in range(10):
    print(f"Processing: {sim_ind+1}")
    np.random.seed(sim_ind)

    # Number of observations
    n = 1000
    # Number of variables
    p = 10
    # Number of relevant variables
    n_signal = 2
    # Signal-to-noise ratio
    snr = 4
    # Correlation coefficient
    rho = 0.8
    # Nominal false positive rate
    alpha = 5e-2

    X, y, _, __ = multivariate_1D_simulation(
        n_samples=n, n_features=p, support_size=n_signal, rho=rho, seed=sim_ind
    )

    # Applying a reLu function on the outcome y to get non-linear relationships
    y = np.maximum(0.0, y)

    ## dcrt Lasso ##
    results_lasso = dcrt_zero(X, y, screening=False, verbose=True)
    typeI_error["Lasso"].append(
        sum(results_lasso[1][n_signal:] < alpha) / (p - n_signal)
    )
    power["Lasso"].append(sum(results_lasso[1][:n_signal] < alpha) / (n_signal))

    ## dcrt Random Forest ##
    results_forest = dcrt_zero(
        X, y, screening=False, statistic="randomforest", verbose=True
    )
    typeI_error["Forest"].append(
        sum(results_forest[1][n_signal:] < alpha) / (p - n_signal)
    )
    power["Forest"].append(sum(results_forest[1][:n_signal] < alpha) / (n_signal))

fig, ax = plt.subplots()
ax.set_title("Type-I Error")
ax.boxplot(typeI_error.values())
ax.set_xticklabels(typeI_error.keys())
ax.axhline(linewidth=1, color="r")

fig, ax = plt.subplots()
ax.set_title("Power")
ax.boxplot(power.values())
ax.set_xticklabels(power.keys())

plt.show()
