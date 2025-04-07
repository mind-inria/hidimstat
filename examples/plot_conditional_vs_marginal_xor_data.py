"""
Conditional vs Marginal Importance on the XOR dataset
===============================================

This example illustrates on XOR data that variables can be conditionally important even
if they are not marginally important. The conditional importance is computed using the
CPI class and the marginal importance is computed using univariate models.
"""

#############################################################################
# To sovlve the XOR problem, we will use a SVC with RBF kernel. The decision function of
# the fitted model shows that the model is able to separate the two classes.

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.base import clone
from sklearn.linear_model import RidgeCV
from sklearn.metrics import explained_variance_score, hinge_loss
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC

from hidimstat import CPI

rng = np.random.RandomState(0)
X = rng.randn(400, 2)
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)

xx, yy = np.meshgrid(
    np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100),
    np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100),
)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
model = SVC(kernel="rbf", random_state=0)
model.fit(X_train, y_train)

Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
fig, ax = plt.subplots()
ax.contourf(
    xx,
    yy,
    Z.reshape(xx.shape),
    levels=20,
    cmap="RdYlBu_r",
    alpha=0.5,
)
sns.scatterplot(
    x=X[:, 0],
    y=X[:, 1],
    hue=Y,
    ax=ax,
    palette="muted",
)
ax.axis("off")
ax.legend(
    title="Class",
)
ax.set_title("Decision function of SVC with RBF kernel")
plt.show()


###############################################################################
# Computing the conditional and marginal importance
# -----------------------------------------------------
# We first compute the marginal importance by fitting univariate models on each feature.
# Then, we compute the conditional importance using the CPI class. The univarariate
# models don't perform above chance, since solving the XOR problem requires to use both
# features. The conditional importance, on the other hand, reveals that both features
# are important (therefore rejecting the null hypothesis $Y \perp\!\!\!\perp X^1 | X^2$)
# .

cv = KFold(n_splits=5, shuffle=True, random_state=0)
clf = SVC(kernel="rbf", random_state=0)
# Compute marginal importance using univariate models
marginal_scores = []
for i in range(X.shape[1]):
    feat_scores = []
    for train_index, test_index in cv.split(X_train):
        X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
        y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]

        X_train_univariate = X_train_cv[:, i].reshape(-1, 1)
        X_test_univariate = X_test_cv[:, i].reshape(-1, 1)

        univariate_model = clone(clf)
        univariate_model.fit(X_train_univariate, y_train_cv)

        feat_scores.append(univariate_model.score(X_test_univariate, y_test_cv))
    marginal_scores.append(feat_scores)


importances = []
for i, (train_index, test_index) in enumerate(cv.split(X_train)):
    X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
    y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]

    clf_c = clone(clf)
    clf_c.fit(X_train_cv, y_train_cv)

    vim = CPI(
        estimator=clf_c,
        method="decision_function",
        loss=hinge_loss,
        imputation_model_continuous=RidgeCV(np.logspace(-3, 3, 10)),
        n_permutations=50,
        random_state=0,
    )
    vim.fit(X_train_cv, y_train_cv)
    importances.append(vim.score(X_test_cv, y_test_cv)["importance"])
importances = np.array(importances).T
# %%
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(5, 2))

# Marginal scores boxplot
sns.boxplot(
    data=np.array(marginal_scores).T,
    orient="h",
    ax=axes[0],
    fill=False,
    color="C0",
    linewidth=3,
)
axes[0].set_xlabel("Marginal Scores (accuracy)")
axes[0].set_ylabel("Features")
axes[0].axvline(x=0.5, color="k", linestyle="--", lw=3)
axes[0].set_ylabel("")
axes[0].set_yticklabels(["X1", "X2"])

# Importances boxplot
sns.boxplot(
    data=np.array(importances).T,
    orient="h",
    ax=axes[1],
    fill=False,
    color="C0",
    linewidth=3,
)
axes[1].axvline(x=0.0, color="k", linestyle="--", lw=3)
axes[1].set_xlabel("Conditional Importance")

sns.despine(ax=axes[1])
sns.despine(ax=axes[0])
plt.show()
