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

def add_metrics_box(ax, corr, r2_feat, loc="upper left"):
    text_str = f"Corr = {corr:.3f}\nR² = {r2_feat:.3f}"

    bbox_props = {
        "boxstyle": "round,pad=0.5",
        "facecolor": "white",
        "alpha": 0.85,
        "edgecolor": "lightgray",
    }

    if loc == "upper left":
        x, y, ha, va = 0.03, 0.95, "left", "top"
    elif loc == "upper right":
        x, y, ha, va = 0.97, 0.95, "right", "top"

    ax.text(
        x,
        y,
        text_str,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment=va,
        horizontalalignment=ha,
        bbox=bbox_props,
    )

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
    X_train_minus = X_train.drop(columns=[feature])
    X_test_minus = X_test.drop(columns=[feature])

    y_feat_train = X_train[feature]
    y_feat_test = X_test[feature]

    variable = RandomForestRegressor(n_estimators=20, random_state=42)
    variable.fit(X_train_minus, y_feat_train)

    y_feat_pred = variable.predict(X_test_minus)
    corr = np.corrcoef(y_feat_test, y_feat_pred)[0, 1]
    r2_feat = variable.score(X_test_minus, y_feat_test)


    display = ALE(model, feature_names=X.columns)
    ax_ale = display.plot(
        X_test,
        X.columns.get_loc(feature),
        grid_resolution=50,
        confidence_interval=True,
    )
    add_metrics_box(ax_ale[1], corr, r2_feat)
    plt.show()


    cpiplot = CPIPlot(model, feature_names=X.columns)
    ax_cpi = cpiplot.plot(
        X_test, X.columns.get_loc(feature), grid_resolution=50, version=0
    )
    add_metrics_box(ax_cpi[1], corr, r2_feat)
    plt.show()


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
    X_train_minus = X_train.drop(columns=[feature])
    X_test_minus = X_test.drop(columns=[feature])

    y_feat_train = X_train[feature]
    y_feat_test = X_test[feature]

    variable = RandomForestRegressor(n_estimators=20, random_state=42)
    variable.fit(X_train_minus, y_feat_train)

    y_feat_pred = variable.predict(X_test_minus)
    corr = np.corrcoef(y_feat_test, y_feat_pred)[0, 1]
    r2_feat = variable.score(X_test_minus, y_feat_test)

    display = ALE(model, feature_names=X.columns)
    ax_ale = display.plot(
        X_test,
        X.columns.get_loc(feature),
        grid_resolution=50,
        confidence_interval=True,
    )
    add_metrics_box(ax_ale[1], corr, r2_feat)
    plt.show()

    cpiplot = CPIPlot(model, feature_names=X.columns)
    ax_cpi = cpiplot.plot(
        X_test, X.columns.get_loc(feature), grid_resolution=50, version=1
    )
    add_metrics_box(ax_cpi[1], corr, r2_feat)
    plt.show()

# %%

# V0 :
# * +Corr +R² => :c
# * +Corr -R² => Légère allure
#
# V1 :
# * +Corr +R² => :D
# * +Corr -R² => :D avec bruit
# * -Corr -R² => :| bruit
