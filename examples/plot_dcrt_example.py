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
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

from hidimstat import D0CRT
from hidimstat._utils.scenario import multivariate_simulation

#############################################################################
# Processing the computations
# ---------------------------

results_list = []
for sim_ind in range(10):
    print(f"Processing: {sim_ind+1}")
    np.random.seed(sim_ind)

    # Number of observations
    n = 100
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
    d0crt_lasso = D0CRT(
        estimator=LassoCV(random_state=42, n_jobs=1), screening_threshold=None
    )
    d0crt_lasso.fit_importance(X, y)
    pvals_lasso = d0crt_lasso.pvalues_
    results_list.append(
        {
            "model": "Lasso",
            "type-1 error": sum(pvals_lasso[np.logical_not(beta_true)] < alpha)
            / (p - n_signal),
            "power": sum(pvals_lasso[beta_true] < alpha) / (n_signal),
        }
    )

    ## dcrt Random Forest ##
    d0crt_random_forest = D0CRT(
        estimator=RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=1,
        ),
        screening_threshold=None,
    )
    d0crt_random_forest.fit_importance(X, y)
    pvals_forest = d0crt_random_forest.pvalues_
    results_list.append(
        {
            "model": "RF",
            "type-1 error": sum(pvals_forest[np.logical_not(beta_true)] < alpha)
            / (n_signal),
            "power": sum(pvals_forest[beta_true] < alpha) / (n_signal),
        }
    )

#############################################################################
# Plotting the comparison
# -----------------------

df_plot = pd.DataFrame(results_list)

_, ax = plt.subplots(nrows=1, ncols=2)
sns.swarmplot(data=df_plot, x="model", y="type-1 error", ax=ax[0], hue="model")
ax[0].axhline(
    alpha,
    linewidth=1,
    color="tab:red",
    ls="--",
    label="Nominal Level",
)
ax[0].legend()
ax[0].set_ylim(-0.01)

sns.boxplot(data=df_plot, x="model", y="power", ax=ax[1], hue="model")

sns.despine()
plt.show()

#############################################################################
# Both methods empirically control the type-I error. In addition, it can
# be observed that the power of the Random Forest model is generally higher
# than that of the Lasso model, indicating a better ability to detect true
# signals in the data. This is likely due to the capacity of the Random
# Forest model to capture interactions and non-linear relationships which
# are introduced in this simulation by the ReLU transformation applied to the
# outcome.
