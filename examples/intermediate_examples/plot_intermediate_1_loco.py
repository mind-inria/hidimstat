"""
Leave-One-Covariate-Out (LOCO) feature importance
=================================================

This example gives a few insights on the LOCO feature importance [:footcite:t:`Williamson_General_2023`],
and compares it across different predictive models on the same regression dataset.
LOCO is model-agnostic and can be applied to any predictive model.
Here, we use a linear model, a random forest, a neural network, and a support vector machine.
We compare the models based on their predictive performance (R2 score) and the LOCO feature importance they yield.
"""

# %%
# How LOCO works
# ---------------
# As the name tells, the LOCO method measures the importance of a feature by evaluating a model that has been
# refit without that feature. The exact measure is the difference in loss :math:`l` between the full model's predictions
# :math:`\hat{f}` and the model retrained without feature :math:`j` :math:`\hat{f}_{-j}`, computed on test data:
#
# .. math::
#
#    LOCO(j)=\frac{1}{n_{\text{test}}}\sum_{i=1}^{n_{\text{test}}}
#    \left[
#      l\left(\hat{f}_{-j}\left(x_i^{-j}\right),y_i\right)
#      -
#      l\left(\hat{f}\left(x_i\right),y_i\right)
#    \right]
#
# LOCO is model-agnostic, meaning that it can be applied to any type of model. However, the model has to be refit for each feature
# to assess its importance which makes this method very computational intensive.

# %%
# Loading and preparing the data
# ------------------------------
# In this example, we use the diabetes dataset for regression and
# evaluate how the different models identify important features.
# The data is then split into training and test sets. These sets are used both to fit
# the predictive models and within the LOCO procedure, which refits models on subsets
# of features that exclude the feature of interest.

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

dataset = load_diabetes()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    shuffle=True,
)

# %%
# Fitting models and computing LOCO feature importance
# ----------------------------------------------------
# We define a list of predictive models to compare. We use RidgeCV for linear
# regression, RandomForestRegressor for a tree-based model, MLPRegressor for a
# neural network, and SVR for a support vector machine, with RBF kernel. We then fit
# each model on the training data, compute the LOCO feature importance on the test
# data, and store the results in a DataFrame for comparison.

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from hidimstat import LOCO

models_list = [
    RidgeCV(),
    RandomForestRegressor(n_estimators=150, random_state=0),
    MLPRegressor(
        hidden_layer_sizes=(8),
        random_state=0,
        max_iter=500,
        learning_rate_init=0.1,
    ),
    SVR(kernel="linear"),
]

df_list = []
for model in models_list:
    # Fit the full model
    model = model.fit(X_train, y_train)
    loco = LOCO(model)
    # With this function call, the method refits the model on all features except one,
    # then computes feature importance. This process is repeated for all features to assess their individual
    # contributions.
    loco.fit(X_train, y_train)
    importances = loco.importance(X_test, y_test)
    df_list.append(
        pd.DataFrame(
            {
                "feature": dataset.feature_names,
                "importance": importances,
                "model": model.__class__.__name__,
                "R2 score": model.score(X_test, y_test),
            }
        )
    )

df_plot = pd.concat(df_list)

# %%
# Visualization of LOCO feature importance
# ----------------------------------------
# Finally, we visualize the LOCO feature importance for each model using a horizontal
# bar plot. The true support features are highlighted in the plot with a green shaded
# background.

import matplotlib.pyplot as plt
import seaborn as sns

ax = sns.barplot(
    data=df_plot,
    y="feature",
    x="importance",
    hue="model",
    palette="muted",
    orient="h",
)
sns.despine()
plt.show()

# %%
# The plot shows that the different models all identify the true support features and
# assign them higher importance scores. However, the magnitude of the importance scores
# varies across models. It can be observed that models with a greater predictive
# performance (higher R2 score) tend to assign higher importance scores.

# %%
# Takeaways
# ---------
# LOCO is a rather intuitive method to assess feature importance, by assessing the contribution
# of single feature through difference in loss between a full model and a retrained model without that feature.
# Here are a few things to pay attention to when using this method:
#
# - LOCO indicates how the model performance reacts to removing features individually, so beware
#   of the interpretation and make sure not to extrapolate.
#
# - LOCO importance of strongly correlated features is always low, since LOCO importance is to
#   be interpreted by the information provided by the other features.
#
# - LOCO is computational expensive, make sure to use it in scenarios where model refitting is not too long.
#
# - LOCO is sensitive to model performance, and therefore can be an unstable feature importance metric.

# %%
# References
# ----------
# .. footbibliography::
