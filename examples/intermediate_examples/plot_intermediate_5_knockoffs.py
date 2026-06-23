"""
Model-X Knockoffs
=============================================================================
Model-X Knockoffs (MXKO) are closely related to the CRT/dCRT family, but they
target a somewhat different goal. Rather than testing one variable at a time,
they create synthetic "knockoff" copies of the features and compare how much
the model prefers the real feature over its knockoff. The main attraction is
feature selection with rigorous false discovery rate (FDR) control.

In this example, we compare feature selection between MXKO and Desparsified
Lasso, then delve into the randomness of Knockoffs and show how to mitigate
this.
"""

# %%
# How Model-X Knockoffs work
# --------------------------
# For each feature :math:`X^j`, construct a knockoff :math:`\tilde{X}^j` such that it has the same correlation
# structure as :math:`X^j` to other features while being independent from :math:`Y`. Swapping :math:`X^j` and
# :math:`\tilde{X}^j` should not change the joint distribution. Then fit a model using both :math:`X` and
# :math:`\tilde{X}`. If the original feature is truly informative, it should consistently outperform its
# knockoff. If not, the model should be unable to distinguish them. Let's start with a basic example.


# %%
# Loading data and fitting variable importance methods
# ----------------------------------------------------
# First, we start by using a simulated regression dataset from sklearn, to generate noisy data,
# and perform a feature selection with the MXKO.

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.datasets import make_regression
from sklearn.linear_model import LassoCV

from hidimstat.knockoffs import ModelXKnockoff
from hidimstat.samplers import GaussianKnockoffs

rng = np.random.default_rng(0)
target_fdr = 0.2
n = 300
p = 100
p_informative = 10

X_, y_, coef = make_regression(
    n_samples=n,
    n_features=p,
    n_informative=p_informative,
    noise=1.0,
    coef=True,
    random_state=42,
)
feature_names = [str(f"Feat.{i}") for i in range(p)]
true_support = np.where(coef != 0)[0]

mx_ko = ModelXKnockoff(
    ko_generator=GaussianKnockoffs(
        cov_estimator=LedoitWolf(assume_centered=True), tol=1e-15
    ),
    estimator=LassoCV(
        max_iter=1000, tol=0.0001, eps=0.01, fit_intercept=False
    ),
    random_state=0,
    preconfigure_lasso_path=False,
    n_jobs=5,
)


ko_importance = mx_ko.fit_importance(X_, y_)
ko_selection = mx_ko.fdr_selection(fdr=target_fdr)

# %%
# Let's plot and compare :

import matplotlib.pyplot as plt
import seaborn as sns

selected_mask_ko = np.array(["not selected"] * len(ko_importance[0]))
selected_mask_ko[ko_selection] = "selected"
df_ko = pd.DataFrame(
    {
        "score": ko_importance[0],
        "variable": feature_names,
        "selected": selected_mask_ko,
    }
)
df_ko = df_ko.sort_values(by=["score"], ascending=False).head(25)

_, ax = plt.subplots()
sns.scatterplot(
    data=df_ko,
    x="score",
    y="variable",
    hue="selected",
    ax=ax,
    palette={"selected": "tab:red", "not selected": "tab:gray"},
)
ax.axvline(
    x=mx_ko.threshold_fdr_,
    color="k",
    linestyle="--",
    label="Threshold",
)
ax.legend()
ax.set_xlabel("KO statistic (LCD)")
ax.set_ylabel("")
ax.set_title("Knockoffs", fontweight="bold")
plt.tight_layout()
plt.show()


# %%
# Randomness in Knockoffs
# -----------------------
# A limitation of Model-X knockoffs is the random generation of the
# knockoff variables, which depends on an accurate estimate of the
# joint distribution of the predictors. Different knockoff realizations
# can lead to variations in the computed importance scores and,
# for weak signals or highly correlated features, different variable
# selections. Moreover, if the feature distribution is misspecified,
# the method may lose power and its theoretical guarantees can be
# weakened. As a result, the stability and performance of Model-X
# knockoffs rely on the quality of the knockoff construction.
# To illustrate this, we duplicate for the true support variables and
# introduce a bit of noise to create correlates.

repeats_noise = 3
rho = 0.9999

noisy_data = [X_]
feature_names_noise = list(feature_names)
for k in range(repeats_noise):
    noisy_data.append(
        rho * X_[:, true_support]
        + np.sqrt(1 - rho**2) * rng.standard_normal((n, p_informative))
    )
    feature_names_noise += [
        f"duplicate #{k}_" + feature_names[i] for i in range(p_informative)
    ]

noisy_data = np.concatenate(noisy_data, axis=1)
all_importances = []

for seed in range(50):
    mx_ko = ModelXKnockoff(
        ko_generator=GaussianKnockoffs(
            cov_estimator=LedoitWolf(assume_centered=True), tol=1e-15
        ),
        estimator=LassoCV(
            max_iter=1000, tol=0.0001, eps=0.01, fit_intercept=False
        ),
        random_state=seed,
        preconfigure_lasso_path=False,
        n_jobs=5,
    )
    mx_ko.fit_importance(noisy_data, y_)
    all_importances.append(mx_ko.importances_[0])

# %%
# We now plot the importance relative variability.

importance_mean = np.array(all_importances).mean(axis=0)
importance_std = np.array(all_importances).std(axis=0)

_, ax1 = plt.subplots(figsize=(10, 4))
ax1.bar(
    np.arange(len(importance_mean)),
    importance_std / (np.abs(importance_mean) + 1e-8),
    capsize=2,
    alpha=0.8,
    label="Mean importance",
)

for i, support in enumerate(coef):
    if support != 0:
        ax1.axvspan(
            i - 0.45,
            i + 0.45,
            color="tab:olive",
            alpha=0.3,
            zorder=-1,
            label="True Support" if i == 8 else None,
        )

ax1.legend(loc="upper right")
plt.xlabel("Feature index")
plt.ylabel("Std / |mean|")
plt.title("Relative std of importance across knockoff runs")
plt.show()

# %%
# We can see that feature importance vary across several iterations, and that
# correlates (feature index > 100) exhibit an important variability, compared to
# the true support features. This is due to the fact that the method measures
# information gain for a feature compared to their knockoff counterpart, which
# makes this method sensitive to correlated features.

# %%
# Fixing the lottery
# ------------------
# Because gambling is not a healthy hobby, to overcome the variability and randomness,
# we generate multiple sets of knockoffs and use aggregation techniques to provide
# a more robust variable importance estimation by adding the argument `n_repeats`,
# such as follows:

mx_ko = ModelXKnockoff(
    ko_generator=GaussianKnockoffs(
        cov_estimator=LedoitWolf(assume_centered=True), tol=1e-15
    ),
    estimator=LassoCV(
        max_iter=1000, tol=0.0001, eps=0.01, fit_intercept=False
    ),
    random_state=42,
    preconfigure_lasso_path=False,
    n_repeats=15,
    n_jobs=5,
)

# %%
# Takeaways
# ---------
# Model-X Knockoffs is a good method for variable selection as it provides
# rigorous false discovery rate (FDR) control, which leads to a conservative
# features selection, and works with any prediction model.
# Unlike CRT-style methods that often test variables individually, knockoffs
# naturally handle many variables at once, which can be practical at high dimension.
# This method is useful to reveal unique information captured by features. It comes
# with a few downsides. The knockoffs generation process can be difficult as the
# joint distribution of X can be difficult for high-dimensional or complex mixed
# data, and the method's power and FDR control depend heavily on knockoff quality.
# Correlated features may compete with each other as we saw in this example. Finally,
# its interpretation is less intuitive than other variable importance estimations, since
# measures whether a variable more informative than a statistically matched fake one
# while controlling FDR.
