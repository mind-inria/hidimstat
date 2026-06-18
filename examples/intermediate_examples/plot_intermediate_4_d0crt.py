"""
Distilled Conditional Randomization Test (dCRT)
==================================================================================

"""

# %%
# A quick introduction to dCRT
# ------------------------------------
# The Distilled Conditional Randomization Test is a method for variable selection that
# tests whether each feature :math:`X^j` is conditionally independent of the target
# :math:`Y` given all other features :math:`X^{-j}`:
#
# .. math::
#
#   \mathcal{H}_0^j: Y \perp\!\!\!\!\perp X^j \mid X^{-j}
#
# The dCRT produces a p-value for each individual feature. This allows conditional
# independence testing at the single-feature level with type-I error control.
# The general procedure is to perform two model distillations on resampled data to build
# the null distribution: one for a regression of :math:`X^j` on :math:`X^{-j}`, and one
# for the regression of :math:`Y` on :math:`X^{-j}`.
# From the residuals obtained from both distillations, we constitute a test statistic
# which is the normalized correlation between the two residuals. From there, p-values
# can be computed.
#
# In HiDimStat's implementation, the X-distillation is set to use a linear model. The
# Y-distillation process can be whichever supervised learner the user wants.
# In this example, we will compare the performance of several estimators in this context.

# %%
# Loading data
# ------------
# We load the diabetes dataset for a regression task

import numpy as np
from sklearn.datasets import load_diabetes

# Setting the seed for reproducibility
rng = np.random.default_rng(0)
dataset = load_diabetes()
X, y = dataset.data, dataset.target

# %%
# Comparing methods
# -----------------
# For this example, we use 3 different models for the Y-distillation:
# a Lasso regressor, a RandomForestRegressor, and a Multi-Layer Perceptron
# regressor. For each of these method, we fit and compute feature importance
# with dCRT.

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPRegressor

from hidimstat import D0CRT

models_list = [
    LassoCV(),
    RandomForestRegressor(n_estimators=150, random_state=0),
    MLPRegressor(
        hidden_layer_sizes=(8),
        random_state=0,
        max_iter=500,
        learning_rate_init=0.1,
    ),
]
df_list = []

for model in models_list:
    d0crt = D0CRT(
        estimator=model,
        screening_threshold=None,
        random_state=42,
    )
    # With this function call, the method fits a model distillation to regress Xj on. X-j,
    # then computes feature importance. This process is repeated for all features to assess their individual
    # contributions.
    importances = d0crt.fit_importance(X, y)
    df_list.append(
        pd.DataFrame(
            {
                "feature": dataset.feature_names,
                "importance": importances,
                "model": model.__class__.__name__,
                "pvalues": d0crt.pvalues_,
            }
        )
    )

# %%
# Plotting
# --------
# We visually compare the feature importance and assimilated p-values
# for each model.

import matplotlib.pyplot as plt
import seaborn as sns

df_plot = pd.concat(df_list)
# Plotting importances and p-values.
fig, axes = plt.subplots(1, 2, sharey="row")
sns.barplot(
    data=df_plot,
    y="feature",
    x="importance",
    hue="model",
    palette="muted",
    orient="h",
    ax=axes[0],
)
sns.barplot(
    data=df_plot,
    y="feature",
    x="pvalues",
    hue="model",
    palette="muted",
    orient="h",
    ax=axes[1],
)
# Adding a vertical line for p<0.05
axes[1].axvline(x=0.05, color="black", linestyle=":", linewidth=1.5)
sns.despine()
handles, labels = axes[0].get_legend_handles_labels()
for ax in axes:
    ax.legend().remove()
fig.legend(
    handles=handles,
    labels=labels,
    loc="center",
    bbox_to_anchor=(0.5, 0.96),
    ncol=3,
)
plt.tight_layout()
plt.show()

# %%
# We see that the variables `sex`, `bmi`, `bp`, and `s5` are considered
# statistically significantly important (p<0.05) for all 3 estimators.
# However, we observe that `s1` and `s2` are deemed important with the
# MLP but not other estimators. While importance tends to be of the same
# order of magnitude across estimators, we can see a greater variability
# for p-values.

# %%
# Takeaways
# ---------
# dCRT produces p-values with finite-sample validity under relatively mild assumptions,
# and conveniently works with any supervised learner. As the name tells, dCRT also preserves
# the relationships between features as it samples from the conditional distribution
# :math:`X^j|X^{-j}`. Finally, the feature selection is executed with type-I error control.
#
# There are a few downsides that should be noted. dCRT does a lot of resampling,
# statistic distribution computation, and X-distillation needs to be executed for each
# feature, which can be computationally intensive and long as the number of feature grows.
# The power heavily depends on the conditional model, which can become very conservative.
# If the conditional distribution is poorly estimated, p-values can become large and true
# signals are missed, which can become a practical bottleneck.
