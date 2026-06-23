"""
General pipeline to assess feature importance
==================================================================================
This example demonstrates how to measure feature importance using hidimstat, and the
general pipeline of functions to call when doing so.
The functions used in the following are generic to any of the feature importance assessment
methods existing in the library. Here, we will use the Conditional Feature Importance
(CFI) [:footcite:t:`Chamma_NeurIPS2023`] on a simulated regression dataset.
"""

# %%
# Loading and preparing the data
# ------------------------------
# We begin by simulating a regression dataset with 10 features, 5 of which
# are in the support set, meaning they contribute to generating the outcome. In this example,
# we use a simulated dataset to have access to the true support set of features and
# evaluate how well the different models identify these important features.
# The data is then split into training and test sets. These sets are used both to fit
# the predictive models and within the LOCO procedure, which refits models on subsets
# of features that exclude the feature of interest.

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y, beta = make_regression(
    n_samples=300,
    n_features=10,
    n_informative=5,
    random_state=0,
    coef=True,
    noise=10.0,
)

# We convert the coefficients of the data-generating process into a binary array
# indicating the true support set of features.
beta = beta != 0

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    shuffle=True,
)

# %%
# Fitting the model and computing feature importance
# ------------------------------------------------------
# To solve the classification task, we use a pipeline that first standardizes the features with StandardScaler,
# followed by a neural network (MLPClassifier) with one hidden layer of 8 neurons.
# Before measuring feature importance, we evaluate the estimator's performance by reporting its :math:`R^2` score.

from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(n_estimators=150, random_state=0)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Accuracy: {clf.score(X_test, y_test):.3f}")

# %%
# Next, we use the CFI class to measure feature importance. Here, we use a RidgeCV
# model to estimate the conditional expectation :math:`\mathbb{E}[X^j | X^{-j}]`.
# Since this is a regression task, we use mean_squared_error.

from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

from hidimstat import CFI

cfi = CFI(
    estimator=clf,
    loss=mean_squared_error,
    imputation_model_continuous=RidgeCV(),
    features_groups={f"Feat {i}": [i] for i in range(X.shape[1])},
    random_state=0,
)
cfi.fit(
    X_train,
    y_train,
)
importances = cfi.importance(X_test, y_test)

# %%
# An alternative is to call the `fit_importance` method which conveniently combines both functions.
# Please keep in mind that both the fit and feature importance will subsequently use the same data.


# %%
# Visualization of CFI feature importance and feature selection
# ----------------------------------------------------------------
# Finally, we visualize the importance of each feature using a bar plot
# thanks to a built-in importance visualization function, and perform
# feature selection with statistical guarantees.

import matplotlib.pyplot as plt
import numpy as np

_, ax = plt.subplots(figsize=(6, 3))
ax = cfi.plot_importance(ax=ax)
ax.set_xlabel("Feature Importance")

# Since the figure displays importance measures in a descending order,
# we need to sort betas accordingly:

sorted_beta = beta[np.argsort(importances)[::-1]]

for i, support in enumerate(sorted_beta):
    if support != 0:
        ax.axhspan(
            i - 0.45,
            i + 0.45,
            color="tab:olive",
            alpha=0.3,
            zorder=-1,
            label="True Support" if i == 1 else None,
        )
ax.legend()

plt.tight_layout()
plt.show()


# %%
# Next, we can make a function call to select the most important features. This can be done through
# one of three difference control mechanisms: through p-values, on False Discovery Rate (FDR)
# or on Family-Wise Error Rate (FWER). Each function returns a boolean mask that indicates us
# which features to keep based on the selected control mechanism.
# Here, we will use select importance through p_values and set a standard threshold of 0.05.

selection = cfi.pvalue_selection(threshold_max=0.05)
important_features = X[:, selection]

# %%
# This is the basic pipeline for feature importance assessment and feature selection.
# We demonstrated the general workflow using CFI, but other implemented methods can be used
# in a similar manner.


# %%
# References
# ----------
# .. footbibliography::
