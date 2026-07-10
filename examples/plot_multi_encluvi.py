"""
Ensemble Clustered Variable Importance
========================================================

This example demonstrates how to measure feature importance using EnCluVI [:footcite:t:`Chamma_NeurIPS2023`] on the wine dataset.
The data are the results of chemical analyses of wines grown in the same region in Italy,
derived from three different cultivars. Thirteen features are used to predict three types
of wine, making this a 3-class classification problem. In this example, we show how to
use different variable importance methods with ensemble clustered inference to identify
which variables are most important for solving the classification task with a neural
network classifier.
"""

# %%
# Loading and preparing the data
# ------------------------------
# We start by loading the dataset and splitting it into training and test sets.

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

dataset = load_wine()
X = dataset.data
y = dataset.target
feat_names = dataset.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=0,
    stratify=y,
    shuffle=True,
)

# %%
# Fitting the model and computing feature importance
# ------------------------------------------------------
# To solve the classification task, we use a pipeline that first standardizes the features with StandardScaler,
# followed by a neural network (MLPClassifier) with one hidden layer of 100 neurons.
# Before measuring feature importance, we evaluate the estimator's performance by reporting its accuracy score.

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

clf = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(100),
        random_state=0,
        max_iter=500,
    ),
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Accuracy: {clf.score(X_test, y_test):.3f}")
# %%
#

import numpy as np
from sklearn.metrics import log_loss

from hidimstat import CFI, D0CRT, LOCI, LOCO, PFI, DesparsifiedLasso

vis = [
    PFI(
        estimator=clf,
        method="predict_proba",
        loss=log_loss,
        n_permutations=20,
        random_state=0,
    ),
    CFI(
        estimator=clf,
        method="predict_proba",
        loss=log_loss,
        n_permutations=20,
        random_state=0,
    ),
    DesparsifiedLasso(),
    D0CRT(
        estimator=clf,
        method="predict_proba",
        random_state=0,
    ),
    LOCO(
        estimator=clf,
        method="predict_proba",
        loss=log_loss,
    ),
    LOCI(
        estimator=clf,
        method="predict_proba",
        loss=log_loss,
    ),
]

importances = []
for vi in vis:
    vi.fit(
        X_train,
        y_train,
    )
    importances.append(vi.importance(X_test, y_test))
importances = np.vstack(importances)


# %%
# Visualization of feature importance
# ----------------------------------------
# We visualize the importance of each feature for each method using a bar plot.

import matplotlib.pyplot as plt

nb_vis = len(vis)
n_cols = 3
n_rows = (nb_vis - 1) // n_cols + 1

_, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 3 * n_rows))
for i in range(nb_vis):
    ax = vis[i].plot_importance(ax=axes[i // n_cols, i % n_cols])
    ax.set_xlabel(f"{type(vis[i]).__name__}")
plt.tight_layout()
plt.show()

# %%
# EnCluVI
# -------
# Now we encapsulate each variable importance method with ensemble clustered
# inference, and plot the importance as bar plots.

import time

import matplotlib.pyplot as plt
from sklearn.cluster import FeatureAgglomeration

from hidimstat import EnCluVI

_, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 3 * n_rows))

for i in range(nb_vis):
    print("-" * 50)
    print(f"{vis[i].__class__}")
    print("-" * 50)

    start_time = time.time()

    encluvi = EnCluVI(
        vi_estimator=vis[i],
        clustering=FeatureAgglomeration(n_clusters=X.shape[1] // 4),
        n_bootstraps=50,
        n_jobs=-1,
        random_state=0,
    )
    encluvi.fit(
        X_train,
        y_train,
    )
    encluvi.importance(X_test, y_test)

    exec_time = time.time() - start_time

    ax = vis[i].plot_importance(ax=axes[i // n_cols, i % n_cols])
    ax.set_xlabel(f"{type(vis[i]).__name__} (Exec time {exec_time:.5f}s)")

plt.tight_layout()
plt.show()


# %%
# References
# ----------
# .. footbibliography::
