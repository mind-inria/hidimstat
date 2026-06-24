"""
Visualization with Accumulated Local Effects (ALE)
==================================================

This example demonstrates how to use **Accumulated Local Effects (ALE)**,
as defined by :footcite:t:`apley2020accumulatedlocaleffects`, to interpret
machine learning models within the `hidimstat` library.

ALE plots allow you to examine a model's dependence on a single feature or a pair of
features. Unlike Partial Dependence Plots (PDPs), ALE avoids extrapolation bias when
features are correlated by averaging localized differences in predictions within conditional intervals.
"""

# %%
# Loading the California Housing dataset
# --------------------------------------
# We will use the California Housing dataset. The goal is to predict the median
# house value based on 8 demographic and geographic attributes.
#
# This dataset contains features such as `MedInc` (Median Income), `AveRooms` (Average Rooms),
# and geographical coordinates, which are naturally strongly correlated.

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print(f"Dataset shape: {X.shape}")
print(f"Dataset features: {data.feature_names}")


# %%
# Training a RandomForestRegressor
# --------------------------------
# We train a model to solve the prediction problem. For this example, we will use
# a `RandomForestRegressor` from scikit-learn.

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=92
)

model = RandomForestRegressor(random_state=92)
model.fit(X_train, y_train)

r2_test_score = model.score(X_test, y_test)
print(f"R² score on the test set: {r2_test_score:.2f}")


# %%
# Accumulated Local Effects for a Single Feature
# ----------------------------------------------
# Let's look at `MedInc` (Median Income). Mathematically, the 1D ALE calculates
# the localized differences in predictions as the feature moves across small quantile
# bins, accumulates them, and centers the final curve around zero.
#
# How to Read the Plot:
# ^^^^^^^^^^^^^^^^^^^^^
# - **The Marginal Strip (Top):** Shows the distribution of the data. ALE values are
#   reliable where data is dense, and should be taken with a grain of salt in sparse regions.
# - **The left Y-Axis (ALE Value):** The value represents the relative contribution of that
#   specific feature value compared to the mean prediction of the dataset. For instance,
#   an ALE value of +0.5 means that when `MedInc` is at that level, the model predicts a
#   house price 0.5 units ($50,000) higher than average, exclusively due to `MedInc`'s behavior.
# - **The right Y-Axis (Mean model prediction):** The value corresponds to the centered value
#   of the ALE shifted to the model mean.

from hidimstat.visualization import ALE

ale = ALE(model, feature_names=X.columns)
_ = ale.plot(X_test, features=0, confidence_interval=True)
plt.show()


# %%
# Accumulated Local Effects on a Pair of Features
# -----------------------------------------------
# The 2D ALE isolates pure second-order interaction effects. Here, we plot the
# interaction between `HouseAge` and `MedInc`.
#
# How to Read the Plot:
# ^^^^^^^^^^^^^^^^^^^^^
# Because main effects are completely removed, the values on this plot show only
# the effect from the correlation between the two variables.
#
# - A value of 0 means the two features act completely independently (their
#   combined effect is just the sum of their individual 1D parts).
# - Positive or negative areas indicate that the combined effect is stronger or
#   weaker than what the individual 1D curves would imply.

_ = ale.plot(X, features=[1, 0], grid_resolution=25, cmap="viridis")
plt.show()


# %%
# References
# ----------
# .. footbibliography::
