"""
Distilled Conditional Randomization Test (dCRT) using Lasso vs Random Forest learners
=====================================================================================

This example compares the performance of d0crt based on
the lasso (1) and random forest (2) implementations. The number of
repetitions is set to 10. The metrics used are the type-I error and
the power
"""

#############################################################################
# Imports needed for this script
# ------------------------------

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor

from hidimstat import D0CRT
from hidimstat._utils.scenario import multivariate_simulation

plt.rcParams.update({"font.size": 21})

typeI_error = {"Lasso": [], "Forest": []}
power = {"Lasso": [], "Forest": []}

#############################################################################
# Processing the computations
# ---------------------------

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
    signal_noise_ratio = 4
    # Correlation coefficient
    rho = 0.8
    # Nominal false positive rate
    alpha = 5e-2

    X, y, beta_true, noise = multivariate_simulation(
        n_samples=n,
        n_features=p,
        support_size=n_signal,
        rho=rho,
        signal_noise_ratio=signal_noise_ratio,
        shuffle=True,
        seed=sim_ind,
    )

    # Applying a reLu function on the outcome y to get non-linear relationships
    y = np.maximum(0.0, y)

    ## dcrt Lasso ##
    d0crt_lasso = D0CRT(estimator=LassoCV(random_state=42, n_jobs=1), screening=False)
    d0crt_lasso.fit_importance(X, y)
    pvals_lasso = d0crt_lasso.pvalues_
    typeI_error["Lasso"].append(sum(pvals_lasso[n_signal:] < alpha) / (p - n_signal))
    power["Lasso"].append(sum(pvals_lasso[:n_signal] < alpha) / (n_signal))

    ## dcrt Random Forest ##
    d0crt_random_forest = D0CRT(
        estimator=RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
        screening=False,
    )
    d0crt_random_forest.fit_importance(X, y)
    pvals_forest = d0crt_random_forest.pvalues_
    typeI_error["Forest"].append(sum(pvals_forest[n_signal:] < alpha) / (p - n_signal))
    power["Forest"].append(sum(pvals_forest[:n_signal] < alpha) / (n_signal))

#############################################################################
# Plotting the comparison
# -----------------------

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
ax[0].set_title("Type-I Error")
ax[0].boxplot(typeI_error.values())
ax[0].set_xticklabels(typeI_error.keys())
ax[0].axhline(linewidth=1, color="r")

ax[1].set_title("Power")
ax[1].boxplot(power.values())
ax[1].set_xticklabels(power.keys())
ax[1].set_ylim(0.5, 1)

plt.show()
