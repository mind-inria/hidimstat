"""
Visualization with CPI
=====================

Some tests with the California Housing dataset
"""

# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyALE import ale
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from hidimstat.visualization import ALE, PDP, CPIPlot

# %%
# Test initial version

# 1. Données et modèle
california = fetch_california_housing(as_frame=True)
X, y = california.data, california.target

model = RandomForestRegressor(n_estimators=20, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=92
)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

# 2. ALE vs PDP

for feature in X.columns:
    display = ALE(model, feature_names=X.columns)
    _ = display.plot(
        X_test,
        X.columns.get_loc(feature),
        grid_resolution=50,
        confidence_interval=True,
    )

    cpiplot = CPIPlot(model, feature_names=X.columns)
    _ = cpiplot.plot(
        X_test, X.columns.get_loc(feature), grid_resolution=50, version=0
    )


# %%
# Test second version

# 1. Données et modèle
california = fetch_california_housing(as_frame=True)
X, y = california.data, california.target

model = RandomForestRegressor(n_estimators=20, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=92
)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

# 2. ALE vs PDP

for feature in X.columns:
    display = ALE(model, feature_names=X.columns)
    _ = display.plot(
        X_test,
        X.columns.get_loc(feature),
        grid_resolution=50,
        confidence_interval=True,
    )

    cpiplot = CPIPlot(model, feature_names=X.columns)
    _ = cpiplot.plot(
        X_test, X.columns.get_loc(feature), grid_resolution=50, version=1
    )

# %%
