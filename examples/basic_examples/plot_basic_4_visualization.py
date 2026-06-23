"""
Visualization with HiDimStat
============================

In previous examples, we used the `plot_importance` function to display bar plots
and visualize which variables were considered important for different methods.
This example demonstrates how to create Partial Dependency Plots (PDPs). This
visualization method allows us to examine a model's dependence on a single feature or
a pair of features. The underlying implementation is built upon
sklearn.inspection.partial_dependence, which calculates the dependence by taking the
average response of an estimator across all possible values of the target feature(s).
We'll use the circles dataset from sklearn to illustrate the basic usage.
"""

# %%
# Loading the wine dataset and training a classifier
# --------------------------------------------------
# We start by loading the wine dataset, training a Multi-Layer Perceptron
# classifier (MLP), and checking its accuracy on the test split.

import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

dataset = load_wine()
X, y = dataset.data, dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

model = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(100), random_state=0, max_iter=500),
)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Accuracy on the test set: {accuracy:.2f}")


# %%
# Partial Dependence for an Individual Feature
# --------------------------------------------
# Once the model is fitted, we use the Partial Dependency Plot (PDP) to visualize its
# dependence on a single input feature, for instance on the first one `alcohol`. The
# resulting plot shows the average response over all other features in the dataset
# of the model (on the :math:`y`-axis) for each possible value of the selected feature (on the :math:`x`-axis).
# The set of selected input features are generally small (one or two) due to constraints of
# ease of interpretation. More specifically, the partial function computed on the set of features
# S (one or two as previously said) marginally to other features :math:`x_c` is estimated
# by calculating averages in the training data
# :math:`\hat{f}_S(x_S)=\frac{1}{n}*\sum_{i=1}^n\hat{f}(x_S,x_C^{(i)})`
#
# The plot also includes the marginal distribution of the feature considered along
# the :math:`x`-axis. This feature distribution is essential for identifying
# low-density regions in the data. Model predictions and the estimated partial
# dependence can be less reliable in these regions.


from hidimstat.visualization import PDP

pdp = PDP(model)
_ = pdp.plot(X_test, features=0)


# %%
# Partial Dependence on a Pair of Features
# ----------------------------------------
# We can similarly visualize the dependence of the model on a pair of features,
# for instance the first and tenth feature, resp. `alcohol` and `color_intensity`.
# Here, the partial dependence is encoded by contour lines (level lines) across the 2D plot.
# The marginal distribution for each eature is also represented along the axes
# to help identify regions where the estimated dependence might be unreliable due to
# a low density of training data.

axes = pdp.plot(X_test, features=[0, 9], cmap="RdBu_r")
plt.show()
