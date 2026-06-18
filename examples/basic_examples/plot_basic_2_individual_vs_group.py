"""
Measuring Individual and Group Variable Importance for Classification
======================================================================

In this example, we show on the Iris dataset how to measure variable importance for
classification tasks. This time, we use the Permutation Feature Importance (PFI) method
with a Support Vector Classifier (SVC). We start by measuring the importance of
individual variables and then show how to measure the importance of groups of variables.

To briefly summarize, PFI (Permutation Feature Importance) shuffles the values of
a feature and measures the increase in the loss when predicting (using om the same
full model) on the shuffled data.

"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import RidgeCV
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hidimstat import CFI

# Define the seeds for the reproducibility of the example
rng = np.random.default_rng(0)

# %%
# Load the iris dataset and add a spurious feature
# ------------------------------------------------
# We start by loading the iris dataset as is, and splitting for training and testing.

dataset = load_wine()
X, y = dataset.data, dataset.target
feature_names = np.array(dataset.feature_names)
print(feature_names)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    shuffle=True,
)

# %%
# We use a Multi-Layer Perceptron with one hidden layer of size 100.
# We fit the estimator, compute the feature importance, and use the
# built-in importance plot function.

model = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(100), random_state=0, max_iter=500),
)
model.fit(X_train, y_train)

cfi = CFI(
    estimator=model,
    imputation_model_continuous=RidgeCV(),
    n_permutations=50,
    random_state=2,
    method="predict_proba",
    loss=log_loss,
)

cfi.fit(X_train, y_train)
importance = cfi.importance(X_test, y_test)
selection = cfi.pvalue_selection(threshold_max=0.05)

_, ax = plt.subplots(figsize=(6, 3))
ax = cfi.plot_importance(ax=ax)
ax.set_xlabel("Feature Importance")
ax.set_yticklabels(feature_names[np.argsort(importance)[::-1]])
# ax.text(0, 0, f"Selected features: {feature_names[selection]}")
plt.tight_layout()
plt.show()


# %%
# Measuring the importance of groups of features
# ----------------------------------------------
# In the example above, CFI did not select some features. This is because it
# measures conditional importance, which is the additional independent information a
# feature provides knowing all the other features. When features are highly correlated,
# this additional information decreases, resulting in lower importance rankings. To
# mitigate this issue, we can group correlated features together and measure the
# importance of these feature groups. Here, we regroup variables into the following groups:
# - "acid": `malic_acid` and `proline`
# - "ash": `ash` and `alcalinity of ash`
# - "phenols": `total_phenols`, `flavanoids`, `nonflavanoid_phenols`, and `proanthocyanins`
# - "color": `hue`, and `color_intensity`

features_groups = {
    "acid": [1, 12],
    "ash": [2, 3],
    "phenols": [5, 6, 7, 8],
    "color": [9, 10],
}


cfi = CFI(
    estimator=model,
    imputation_model_continuous=RidgeCV(),
    n_permutations=50,
    random_state=2,
    method="predict_proba",
    loss=log_loss,
    features_groups=features_groups,
)

importance = cfi.fit_importance(X_test, y_test)

_, ax = plt.subplots(figsize=(6, 3))
ax = cfi.plot_importance(ax=ax)
ax.set_xlabel("Feature Importance")
plt.tight_layout()
plt.show()
