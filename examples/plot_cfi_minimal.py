"""
Conditional Feature Importance (CFI) on the wine dataset
========================================================

This example demonstrates how to measure feature importance using CFI on the wine dataset.
The data are the results of chemical analyses of wines grown in the same region in Italy,
derived from three different cultivars. Thirteen features are used to predict three types
of wine, making this a 3-class classification problem. In this example, we show how to
use CFI to identify which variables are most important for solving the classification
task with a neural network classifier.
"""

# %%
from sklearn.datasets import load_wine
from sklearn.linear_model import RidgeCV
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hidimstat import CFI
from hidimstat.visualization import plot_importance

# %%
# We start by loading the dataset and splitting it into training and test sets.
# This split will be used both for training the classifier and for the CFI method.
# The CFI method measures the importance of a feature by generating perturbations
# through sampling from the conditional distribution :math:`p(X^j | X^{-j})`,
# which is estimated on the training set.

X, y = load_wine(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=0,
    stratify=y,
    shuffle=True,
)

# %%
# To solve the classification task, we use a pipeline that first standardizes the features with StandardScaler,
# followed by a neural network (MLPClassifier) with one hidden layer of 100 neurons.
# Before measuring feature importance, we evaluate the estimator's performance by reporting its accuracy score.

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
print("Accuracy:", clf.score(X_test, y_test))
# %%
# Next, we use the CFI class to measure feature importance. Here, we use a RidgeCV
# model to estimate the conditional expectation :math:`\mathbb{E}[X^j | X^{-j}]`.
# Since this is a classification task, we use log_loss and specify the "predict_proba"
# method of our estimator.

cfi = CFI(
    estimator=clf,
    loss=log_loss,
    method="predict_proba",
    imputation_model_continuous=RidgeCV(),
    random_state=0,
)
cfi.fit(X_train, y_train)
importances = cfi.importance(X_test, y_test)


# %%
# Finally, we visualize the importance of each feature using a bar plot.
import matplotlib.pyplot as plt

_, ax = plt.subplots(figsize=(6, 3))
ax = plot_importance(
    importances["importance"],
    feature_names=load_wine().feature_names,
    ax=ax,
)
ax.set_xlabel("Feature Importance")
plt.tight_layout()
