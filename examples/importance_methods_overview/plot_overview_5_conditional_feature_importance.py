"""
Conditional Feature Importance (CFI)
==================================================================================

In this example, we explain how Conditional Feature Importance (CFI) works, and
showcase it on the California housing dataset.
"""

# %%
# Conditional Feature Importance
# ------------------------------
# CFI proposes a way of computing a mean decrease in accuracy, without refitting the model.
# Unlike Permutation Feature Importance (PFI), samples are not blindly permuted,
# but rather we generate samples for feature :math:`x_j` based on its conditional distribution
# of the remaining features :math:`x_{-j}`. Intuitively, he bigger the decrease in loss, the
# more important the feature is for the model. This method has the benefit of being
# model-agnostic. Let's have a closer look at how it works with an example.

# %%
# Loading the California housing dataset
# --------------------------------------
# The California housing dataset is a regression dataset with 8 features. We add a
# spurious feature that is a linear combination of 3 features plus some noise.
# The spurious feature does not provide any additional information about the target.

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Setting the seed for reproducibility
rng = np.random.default_rng(0)
dataset = fetch_california_housing()
X_, y_ = dataset.data, dataset.target
# Only use 2/3 of samples to speed up the example
X, _, y, _ = train_test_split(
    X_,
    y_,
    test_size=0.6667,
    random_state=0,
    shuffle=True,
)

redundant_coef = rng.choice(np.arange(X.shape[1]), size=(3,), replace=False)
X_spurious = X[:, redundant_coef].sum(axis=1)
X_spurious += rng.normal(0, scale=np.std(X_spurious) * 0.5, size=X.shape[0])

X = np.hstack([X, X_spurious[:, np.newaxis]])

feature_names = [*dataset.feature_names, "Spurious"]
print(f"The dataset contains {X.shape[0]} samples and {X.shape[1]} features.")

# Compute the correlation matrix and plot its lower triangle
correlation_matrix = np.corrcoef(X, rowvar=False)
fig, ax = plt.subplots()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(
    correlation_matrix,
    mask=mask,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    square=True,
    cbar_kws={"shrink": 0.8},
    ax=ax,
)
ax.set_title("Correlation Matrix")
ax.set_yticks(np.arange(len(feature_names)) + 0.5)
ax.set_yticklabels(labels=feature_names, fontsize=10, rotation=45)
ax.set_xticks(np.arange(len(feature_names)) + 0.5)
ax.set_xticklabels(labels=feature_names, fontsize=10, rotation=45)
plt.tight_layout()
plt.show()

# %%
# Improvement from PFI
# --------------------
# One of the main pitfalls of PFI is that it leads to extrapolation bias, i.e., it
# forces the model to predict from regions of the feature space that are not present in
# the training data. This can be seen on the California housing dataset, by comparing
# the original latitude and longitude values with the permuted values. Indeed,
# permuting the longitude values leads to generating combinations of latitude and
# longitude that fall outside of the borders of California and therefore are by
# definition not in the training data. If we perform conditional sampling however,
# we observe that we can generate reasonable values of longitude and latitude, as shown
# by the graph below:

from matplotlib.lines import Line2D
from sklearn.linear_model import RidgeCV

from hidimstat.samplers.conditional_sampling import ConditionalSampler

X_train, X_test = train_test_split(
    X,
    test_size=0.3,
    random_state=0,
)

conditional_sampler = ConditionalSampler(
    model_regression=RidgeCV(alphas=np.logspace(-3, 3, 5)),
)

conditional_sampler.fit(X_train[:, :7], X_train[:, 7])
X_test_sample = conditional_sampler.sample(
    X_test[:, :7], X_test[:, 7], n_samples=1, random_state=0
).ravel()

fig, ax = plt.subplots()

sns.histplot(
    x=X_test[:, 6],
    y=X_test[:, 7],
    color="tab:blue",
    ax=ax,
    alpha=0.9,
)
sns.scatterplot(
    x=X_test[:, 6],
    y=X_test_sample,
    ax=ax,
    alpha=0.2,
    c="tab:green",
)
sns.scatterplot(
    x=X_test[:, 6],
    y=rng.permutation(X_test[:, 7]),
    ax=ax,
    alpha=0.2,
    c="tab:orange",
)

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="tab:blue",
        markersize=10,
        label="Original",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="tab:orange",
        markersize=10,
        label="Permutation",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="tab:green",
        markersize=10,
        label="Conditional Permutation",
    ),
]
ax.legend(handles=legend_elements, loc="upper right")
ax.set_ylim(X[:, 7].min() - 0.1, X[:, 7].max() + 0.1)
sns.despine(ax=ax)
ax.set_xlabel("Latitude")
ax.set_ylabel("Longitude")
plt.show()

# %%
# Fitting a predictive model
# --------------------------
# We fit a neural network model to the California housing dataset. CFI is a
# model-agnostic method, we therefore illustrate its behavior when using a neural
# network model.

from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

fitted_estimators = []
scores = []
model = TransformedTargetRegressor(
    regressor=make_pipeline(
        StandardScaler(),
        MLPRegressor(
            random_state=0,
            hidden_layer_sizes=(32, 16, 8),
            early_stopping=True,
            learning_rate_init=0.01,
            n_iter_no_change=5,
        ),
    ),
    transformer=StandardScaler(),
)

# Split the data across the 5 folds, fit, and evaluate the model.
kf = KFold(n_splits=5, shuffle=True, random_state=0)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model_c = clone(model)

    model_c = model_c.fit(X_train, y_train)
    fitted_estimators.append(model_c)
    y_pred = model_c.predict(X_test)
    scores.append(r2_score(y_test, y_pred))

print(
    f"Cross-validation R2 score: {np.mean(scores):.3f} ± {np.std(scores):.3f}"
)

# %%
# Measuring feature importance with CFI
# -------------------------------------
# We use the `PermutationFeatureImportance` class to compute the PFI in a cross-validation
# way. We then derive a p-value from importance scores using a one-sample t-test.

import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.linear_model import RidgeCV

from hidimstat import CFI

conditional_importances = []
for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model_c = fitted_estimators[i]

    # Compute conditional feature importance
    cfi = CFI(
        model_c,
        imputation_model_continuous=RidgeCV(
            alphas=np.logspace(-3, 3, 5),
            cv=KFold(n_splits=3),
        ),
        random_state=0,
        n_jobs=5,
    )
    cfi.fit(X_test, y_test)

    conditional_importances.append(cfi.importance(X_test, y_test))

conditional_importances = np.stack(conditional_importances)
cfi_pval = ttest_1samp(
    conditional_importances, 0.0, axis=0, alternative="greater"
).pvalue

df_pval = pd.DataFrame(
    {
        "pval": cfi_pval,
        "method": ["CFI"] * len(cfi_pval),
        "variable": feature_names,
        "log_pval": -np.log10(cfi_pval),
        "importance": conditional_importances.mean(axis=0),
    }
)

pval_threshold = 0.05

fig, ax = plt.subplots()
sns.barplot(
    data=df_pval,
    x="importance",
    y="variable",
    hue="method",
    palette="muted",
    ax=ax,
)
for i, pval in enumerate(cfi_pval):
    if pval < pval_threshold:
        ax.scatter(
            df_pval.iloc[i].importance + 0.01,
            i,
            color="red",
            marker="*",
            label="pvalue < 0.05" if i == 0 else "",
        )
# ax.axvline(x=-np.log10(pval_threshold), color="red", linestyle="--")
# ax.set_xlabel("-$\\log_{10}(pval)$")
# ax.set_xlabel("Conditional importance")
plt.tight_layout()
plt.show()

# %%
# We can see that spurious feature is not considered important by the model,
# and that correlated features such as latitude and longitude are still considered
# important - although not in the same relative magnitude to other features if we compare
# with PFI from the other example.

# %%
# Takeaways
# ---------
# To sum things up, CFI is a model-agnostic feature importance method that measures a loss decrease when
# doing value permutation across samples for a single feature.
# Contrary to PFI, CFI is designed to take into account the conditional distribution of a feature over others.
# It handles correlated features better than PFI, and successfully reduces bias from spurious features
# and spurious correlations.
# However, CFI can become computationaly intensive since the sample generation mechanism is based on the
# conditioning of all other features, leading to higher costs when the number of features increases.
