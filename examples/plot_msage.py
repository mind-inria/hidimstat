"""
Shapley Additive Global Importance (SAGE) example
=================================================

In this example, we demonstrate how to measure feature importance using
SAGE :footcite:t:`Covert2020` on the diabetes dataset.
Read more in the :ref:`User Guide <shapley_additive_global_explanation>`.
For this example, we use
the marginal version of SAGE, which limits the computational cost. To further
reduce the computational cost, Shapley values are estimated using a Monte Carlo
approximation. Only a subset of all possible feature coalitions is sampled.
This is controlled by the `n_subsets` parameter. Finally, the expectation over
the marginal distribution is also approximated using `n_permutations`.
"""


# %%
# LightGBM example on the bike sharing dataset
# --------------------------------------------
# We demonstrate how to use SAGE on the bike sharing
# dataset. We fit a LightGBM model and compute its :math:`R^2` score on a
# held-out test set.

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

bike_sharing = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True)
df = bike_sharing.frame
df = df[df["year"] == 0].drop(columns=["year"])

X = df.drop(columns=["count"]).to_numpy()
X = OrdinalEncoder().fit_transform(X)
y = df["count"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = HistGradientBoostingRegressor(random_state=0, max_depth=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("R2 score:", r2_score(y_test, y_pred))


# %%
# SAGE feature importance
# -----------------------
# We compute the SAGE feature importance for the fitted model. To keep the
# computational cost tractable, we use a subset of the test set to compute the
# SAGE values. Finally, the SAGE values are plotted using the `plot_importance`
# function.

import matplotlib.pyplot as plt

from hidimstat import SAGE

sage = SAGE(
    model,
    n_subsets=512,
    n_permutations=10,
    random_state=0,
    n_jobs=8,
)
sage.fit(X_train)
subsample_size = 1024
rng = np.random.default_rng(0)
subsample_ids = rng.choice(len(X_test), size=subsample_size, replace=False)
sage.importance(X_test[subsample_ids], y_test[subsample_ids])
ax = sage.plot_importance(
    feature_names=df.drop(columns=["count"]).columns.tolist(),
    color="tab:purple",
)
ax.semilogx()
plt.tight_layout()
plt.show()


# %%
# This analysis reveals that the hour of the day is the most important feature
# for predicting bike sharing demand. This example also illustrates a specific
# property of SAGE: it tends to distribute importance over correlated features.
# This can be seen for the temperature and "feel temperature" features,
# which have a near-perfect correlation, as seen in the correlation matrix below,
# and are both assigned similar importance.
# This property is a consequence of the axioms of Shapley values, and contrasts
# with other feature importance methods, such as
# :class:`~hidimstat.LOCO` or :class:`~hidimstat.CFI`.

corr_mat = np.corrcoef(X_train, rowvar=False)
fig, ax = plt.subplots()
ax.imshow(corr_mat, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(np.arange(corr_mat.shape[0]))
ax.set_yticks(np.arange(corr_mat.shape[0]))
ax.set_xticklabels(
    df.drop(columns=["count"]).columns.tolist(), rotation=45, ha="right"
)
ax.set_yticklabels(df.drop(columns=["count"]).columns.tolist())
cbar = ax.figure.colorbar(
    ax.imshow(corr_mat, cmap="coolwarm", vmin=-1, vmax=1), label="correlation"
)
plt.tight_layout()
plt.show()

# %%
# References
# ----------
# .. footbibliography::
