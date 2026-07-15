"""
Shapley Additive Global Importance (SAGE) example
=================================================

In this example, we demonstrate how to use measure feature importance using
SAGE :footcite:t:`Covert2020` on the diabetes dataset. For this example, we use
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
# dataset. We fit a LightGBM model and compute its prediction $R^2$ score on a
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
sage.plot_importance(
    feature_names=df.drop(columns=["count"]).columns.tolist(),
    color="tab:purple",
)
