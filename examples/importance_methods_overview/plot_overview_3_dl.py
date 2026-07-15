"""
Desparsified Lasso
===================================================
We present and illustrate the limitations of the Desparsified
Lasso (:class:`hidimstat.DesparsifiedLasso`) method through a simulated 2D example where the number of features is higher than
the number of samples. We then present two methods that leverage the data's
spatial structure to build clusters and perform the inference at the cluster level,
with clustered DL (:class:`hidimstat.CluDL`) and with Ensembled CluDL (:class:`hidimstat.EnCluDL`).
"""

# %%
# Feature importance with Desparsified Lasso
# ------------------------------------------
# Desparsified Lasso estimates the regression coefficients obtained
# from a Lasso regression, after correcting the bias induced by the L1-penalty. This method is
# particularly useful in high-dimensional settings where the number of features exceeds the number of samples.
# The feature importance thus corresponds to the estimated coefficients :math:`\hat{\beta}`.
# The aim of the inference step is to recover the support while controlling
# the Family-Wise Error Rate (FWER) at a targeted level of 0.1. To achieve this, we use
# the Bonferroni correction, applying a factor equal to the number of features. For more
# details about the Desparsified Lasso method, see :footcite:t:`javanmard2014confidence`,
# :footcite:t:`zhang2014confidence` and :footcite:t:`van2014asymptotically`.

# %%
# Loading and preparing the data
# ------------------------------
# In this example, we use the diabetes dataset for regression and
# evaluate how the different models identify important features.
# The data is then split into training and test sets. These sets are used both to fit
# the predictive models and within the LOCO procedure, which refits models on subsets
# of features that exclude the feature of interest.

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_diabetes(as_frame=True)
X = data["data"].to_numpy()
y = data["target"].to_numpy()
feature_names = data["data"].columns.tolist()

X = StandardScaler().fit_transform(X)
y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    shuffle=True,
)

# %%
# Add spurious features
# ---------------------
# To evaluate the feature selection capabilities of the Desparsified Lasso, we artificially
# add spurious features. These are constructed as random linear combinations of the original
# features plus noise, ensuring they are correlated with the predictors but have no true
# association with the target variable.

import numpy as np

seed = 0
rng = np.random.default_rng(seed=seed)
n_spurious = 10
X_spurious_list = []
for i in range(n_spurious):
    X_spurious = (
        X[:, rng.choice(X.shape[1], size=3, replace=False)]
        + 1 * rng.normal(size=X[:, :3].shape)
    ).sum(axis=1, keepdims=True)
    X_spurious_normalized = StandardScaler().fit_transform(X_spurious)
    X_spurious_list.append(X_spurious_normalized)
    feature_names.append(f"spurious_{i}")
X = np.hstack([X, *X_spurious_list])

# %%
# Predictive performance benchmark
# --------------------------------
# Before assessing feature importance, we evaluate the predictive performance of the Lasso model
# (that will be used as base estimator in Desparsified Lasso) and a standard Linear Regression
# model using cross-validation. We expect the Lasso to perform better thanks to its regularization
# effect, especially with the added spurious features. We visualize the correlation matrix of the
# features and the distribution of R2 scores.

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import cross_val_score

lasso_model = LassoCV(max_iter=1000)
linear_model = LinearRegression()

cv_score_lasso = cross_val_score(lasso_model, X, y, cv=3)
cv_score_linear = cross_val_score(linear_model, X, y, cv=3)

_, ax = plt.subplots(1, 2, width_ratios=[2, 1], figsize=(7, 4))
corr_mat = data["data"].corr()
sns.heatmap(
    corr_mat,
    cmap="coolwarm",
    ax=ax[0],
    cbar_kws={"label": "Correlation"},
    mask=np.triu(np.ones_like(corr_mat, dtype=bool)),
)
sns.boxplot(data=[cv_score_lasso, cv_score_linear], ax=ax[1])
ax[1].set_xticklabels(["Lasso", "Linear\nRegression"])
ax[1].set_ylabel("R2 score")
sns.despine()
plt.tight_layout()
_ = plt.show()

# %%
# Feature importance with Desparsified Lasso
# ------------------------------------------
# We fit the Desparsified Lasso on the dataset to obtain de-biased coefficient estimates and 95%
# confidence intervals.

import pandas as pd

from hidimstat import DesparsifiedLasso

dl = DesparsifiedLasso(
    estimator=LassoCV(max_iter=1000),
    confidence=0.95,
    model_x=LassoCV(),
    n_jobs=5,
    random_state=seed,
)
dl.fit_importance(X, y)

selected = dl.fdr_selection(fdr=0.1, two_tailed_test=True)
df_plot = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": dl.importances_,
        "selected": selected,
        "lasso_coef": dl.estimator.coef_,
        "confidence_min": dl.confidence_bound_min_,
        "confidence_max": dl.confidence_bound_max_,
    }
)
df_plot = df_plot.sort_values(by="importance", key=np.abs, ascending=False)

# %%
# Results visualization
# ---------------------
# We visualize the de-biased coefficient estimates (circles) with their 95% confidence intervals and
# plot the original Lasso coefficient estimates (triangles) for comparison.
#
# We observe that the confidence intervals help rule out spurious features that the standard Lasso
# might otherwise select. For the non-spurious features, while the Lasso coefficients are shrunk
# towards zero, the Desparsified Lasso provides a correction, often resulting in larger absolute
# coefficient estimates.

from matplotlib.lines import Line2D

# sphinx_gallery_thumbnail_number = 2

_, ax = plt.subplots(figsize=(6, 4))

ax.errorbar(
    x=df_plot["feature"],
    y=df_plot["importance"],
    yerr=[
        df_plot["importance"] - df_plot["confidence_min"],
        df_plot["confidence_max"] - df_plot["importance"],
    ],
    ecolor="gray",
    capsize=8,
    ls="",
)
sns.pointplot(
    data=df_plot,
    x="feature",
    y="importance",
    hue="selected",
    linestyles="",
    palette=["tab:green", "tab:red"],
    markeredgewidth=0.5,
    markeredgecolor="gray",
    markersize=8,
)
sns.pointplot(
    data=df_plot,
    x="feature",
    y="lasso_coef",
    hue=np.abs(df_plot["lasso_coef"]) > 1e-3,
    linestyles="",
    markers="^",
    palette=["tab:orange", "tab:blue"],
    markeredgewidth=0.5,
    markeredgecolor="gray",
)

legend_elements = [
    Line2D(
        [0],
        [0],
        marker=m,
        color=c,
        label=label,
        markersize=8,
        linestyle="",
    )
    for c, label, m in [
        ("tab:green", "Desparsified Lasso selected", "o"),
        ("tab:red", "Desparsified Lasso not selected", "o"),
        ("tab:blue", "Lasso coef $|\\beta| > 0$", "^"),
        ("tab:orange", "Lasso coef $|\\beta| = 0$", "^"),
    ]
]

ax.legend(handles=legend_elements, loc="best")

ax.axhline(y=0, color="k", linestyle="--", linewidth=0.8)
ax.set_xticklabels(df_plot["feature"], rotation=45, ha="right")
ax.set_ylabel("$\\hat{\\beta}$: Coefficient estimates")
ax.set_xlabel("")

sns.despine()
plt.tight_layout()
_ = plt.show()

# %%
# Takeaways
# ---------
# While some spurious features are selected by the Lasso, the Desparsified Lasso provides
# better control over false discoveries. The combination of point estimates and confidence
# intervals allows for both robust feature selection and statistically-grounded feature importance
# quantification.
#
# Desparsified Lasso is a fast feature importance method, that gives an easy
# interpretation of its coefficients as feature importance. However, it assumes
# a linear relationship, which might not capture well real data structure.
