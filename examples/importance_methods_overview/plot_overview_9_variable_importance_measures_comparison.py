"""
Comparing Feature Importance Across Methods
===========================================
We illustrate here that methods implemented in the library such as LOCO, Conditional
Feature Importance (CFI), dCRT and Model-X Knockoffs answer different questions,
live on different scales, and should generally be compared by rank or selection
rather than by raw magnitude.
"""

# %%
# Can we compare methods ?
# ------------------------
# Since the best method may depend on the context, it is important to compare different variable
# importance measures. Indeed, they each answer different questions, and seem to focus on
# different quantity measures. We explore this by explaining the values returned by each method,
# and looking at the importance ranking that each method produces.
#
# We start by defining two functions to generate linear data with an autoregressive structure
# with Toeplitz covariance matrix.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
n_jobs = 1

n_samples = 200
n_features = 10
rho = 0.85

# Toeplitz-like covariance with a couple of strong pairwise correlations
cov = np.eye(n_features)
cov[2, 3] = cov[3, 2] = rho
cov[4, 0] = cov[0, 4] = rho

X = rng.multivariate_normal(mean=np.zeros(n_features), cov=cov, size=n_samples)

beta = np.zeros(n_features)
beta[0] = 2.0  # causal variable is correlated with noisy feature 4
beta[2] = 1.5  # causal variable is correlated with feature 3
beta[3] = 1.5  # causal variable is correlated with feature 2
beta[7] = 1.0  # causal variable and independent

noise = rng.normal(scale=1.0, size=n_samples)
y = X @ beta + noise

feature_names = [f"X{i}" for i in range(n_features)]
true_support = beta != 0
print(
    "True relevant features:",
    [f for f, s in zip(feature_names, true_support, strict=False) if s],
)

# %%
# We now fit a RandomForestRegressor on the dataset.

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

model = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=n_jobs)
model.fit(X_train, y_train)
print("R^2 score:", r2_score(y_test, model.predict(X_test)))

# %%
# Creating the feature importance methods
# ---------------------------------------
# CFI measures the drop in predictive performance loss when we break the dependence between :math:`X_j`
# and :math:`Y` conditionally to the rest, without refitting. The value thus exists in the space of the
# loss function.

from hidimstat import CFI

cfi = CFI(
    estimator=model,
    n_permutations=50,
    n_jobs=n_jobs,
    random_state=0,
)
cfi.fit(X_train, y_train)
cfi_importance = cfi.importance(X_test, y_test)

cfi_importance = pd.Series(cfi_importance, index=feature_names, name="CFI")
print(cfi_importance.sort_values(ascending=False))

# %%
# LOCO refits the model without :math:`X_j` and measures the resulting increase in test loss .
# The values also exists in the space of the loss function, but reflects the model's loss when it
# doesn't have access to :math:`X_j` at all.

from hidimstat import LOCO

loco = LOCO(
    estimator=model,
    n_jobs=n_jobs,
)
loco.fit(X_train, y_train)
loco_importance = loco.importance(X_test, y_test)

loco_importance = pd.Series(loco_importance, index=feature_names, name="LOCO")
print(loco_importance.sort_values(ascending=False))

# %%
# dCRT returns a p-value for the null :math:`X_j \perp Y \mid X_{-j}`. This is a significance measure.
# A smaller p-value means stronger evidence against conditional independence, it does not mean
# a larger effect. Two features can have very different p-values purely due to power differences
# even if their true conditional effect sizes are similar.

from hidimstat import D0CRT

dcrt = D0CRT(
    estimator=model,
    screening_threshold=None,
    random_state=42,
    n_jobs=n_jobs,
)
dcrt.fit(X_train, y_train)
dcrt_importance = dcrt.importance(X_test, y_test)

dcrt_importance = pd.Series(dcrt_importance, index=feature_names, name="dCRT")
print(dcrt_importance.sort_values(ascending=False))

# %%
# Model-X Knockoffs produces a per-feature statistic used to construct a data-dependent
# threshold for FDR-controlled selection. The magnitude of values is not a calibrated effect size,
# it depends on the specifically chosen statistic and the knockoff construction.
# It's designed for selection, not to compare within-method feature importance values.

from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LassoCV

from hidimstat import ModelXKnockoff
from hidimstat.samplers import GaussianKnockoffs

mx_ko = ModelXKnockoff(
    ko_generator=GaussianKnockoffs(
        cov_estimator=LedoitWolf(assume_centered=True), tol=1e-15
    ),
    estimator=LassoCV(
        max_iter=1000, tol=0.0001, eps=0.01, fit_intercept=False
    ),
    random_state=0,
    preconfigure_lasso_path=False,
    n_repeats=15,
    n_jobs=n_jobs,
)

mx_ko.fit(X_train, y_train)
ko_importance = mx_ko.importance(X_test, y_test)

ko_importance = pd.Series(ko_importance[0], index=feature_names, name="MXKO")
print(ko_importance.sort_values(ascending=False))

# %%
# Why you can't just compare raw importance values
# ------------------------------------------------
#
# Let's look at the raw values side by side. The point of this plot isn't the
# specific numbers, it's that the y-axes have no common meaning and comparison ground:
# p-values shrink toward 0, knockoff statistics can be positive or negative, and CPI and
# LOCO live in units of held-out loss.

import matplotlib.pyplot as plt

results = {}
results["CFI"] = cfi_importance
results["LOCO"] = loco_importance
results["dCRT"] = dcrt_importance
results["MXKO"] = ko_importance

results_df = pd.DataFrame(results)
results_df.insert(0, "True support", true_support)
results_df

fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=False)

colors = ["#2d2ae1" if t else "#c91d1d" for t in true_support]
for ax, (name, vals) in zip(axes.flatten(), results.items(), strict=False):
    ax.bar(feature_names, vals.values, color=colors)
    ax.set_title(name)
    ax.tick_params(axis="x", rotation=45)

fig.suptitle("Raw values are not on a common scale.")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Comparing ranks
# ---------------
# Even though the raw numbers aren't on the same scale, do methods agree on which features
# matter most ? Spearman rank correlation is a fair way to compare across methods.

import seaborn as sns

corr = pd.DataFrame(results).corr(method="spearman")

ax = plt.axes()
sns.heatmap(corr, annot=True, vmin=-1, vmax=1, ax=ax, cmap="vlag")
ax.set_title("Spearman rank correlation of important features")
plt.tight_layout()
plt.show()

# %%
# Takeaways
# ---------
# Here are the two takeaways on how we can combine the analysis of different feature importance
# methods:
# - Raw magnitudes were never meant to be compared across methods that we presented here.
# It is important to keep in mind that they target different quantities and answer very
# different questions.
# - Rankings and selection decisions should be the basis for comparison.
#
