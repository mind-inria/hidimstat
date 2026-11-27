"""
Visualization with MPlot
===========================================

This example demonstrates how to create an MPlot.
"""

# %%
# Loading the circles dataset
# ----------------------------
# We start by sampling a synthetic dataset using the `make_circles` function from
# `sklearn.datasets`.

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=500, noise=0.1, factor=0.7, random_state=0)

# Visualizing the dataset
_, ax = plt.subplots()
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, ax=ax)
ax.set_xlabel("X0")
ax.set_ylabel("X1")
sns.despine(ax=ax)
c1 = plt.Circle(
    (0, 0), 0.85, color="k", ls="--", fill=False, label="class boundary"
)
ax.add_patch(c1)
_ = ax.legend(loc="upper right")


# %%
# Training a classifier
# ---------------------
# Next, we train a model to solve the binary classification task presented by the
# non-linearly separable circles dataset. For this example, we'll use a gradient
# boosted tree ensemble, specifically the HistGradientBoostingClassifier from
# scikit-learn.

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)
model = HistGradientBoostingClassifier(random_state=0)

model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)

auc = roc_auc_score(y_true=y_test, y_score=y_pred[:, 1])
print(f"ROC AUC on the test set: {auc:.2f}")


# %%
# MPlotting
# --------------------------------------------
#


from hidimstat.visualization import MPlot

# sphinx_gallery_thumbnail_number = 2
mplot = MPlot(estimator=model)
mplot.plot(X_test, features=0)


# %%
# 2D MPlotting
# ----------------------------------------
#

axes = mplot.plot(X_test, features=[0, 1], cmap="RdBu_r")
c1 = plt.Circle((0, 0), 0.85, color="k", ls="--", fill=False, zorder=10)
_ = axes[1, 0].add_patch(c1)
plt.show()
