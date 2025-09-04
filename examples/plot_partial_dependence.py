"""
Partial Dependence and Individual Conditional Expectation Plots
===============================================================
This example is a modified version of the the example proposed in Scikit-learn:
https://scikit-learn.org/1.7/auto_examples/inspection/plot_partial_dependence.html
Partial dependence plots show the dependence between the target function [1]_
and a set of features of interest, marginalizing over the values of all other
features (the complement features). We limit to only one feature for the
calculation of the importance because the extension of this method to multiple 
features is not trivial.
Similarly, an individual conditional expectation (ICE) plot :footcite:t:`goldstein2015peeking`
shows the dependence between the target function and a feature of interest.
However, unlike partial dependence plots, which show the average effect of the
features of interest, ICE plots visualize the dependence of the prediction on a
feature for each sample separately, with one line per sample.
Only one feature of interest is supported for ICE plots.
This example shows how to obtain partial dependence and ICE plots from a
:class:`~sklearn.neural_network.MLPRegressor` and a
:class:`~sklearn.ensemble.HistGradientBoostingRegressor` trained on the
bike sharing dataset. The example is inspired by :footcite:t:`molnar2025`.

Notes
-----
.. [1] For classification you can think of it as the regression score before
       the link function.

References
----------
.. footbibliography::

"""

# %%
# Bike sharing dataset preprocessing
# ----------------------------------
#
# We will use the bike sharing dataset. The goal is to predict the number of bike
# rentals using weather and season data as well as the datetime information.
from sklearn.datasets import fetch_openml

bikes = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True)
# Make an explicit copy to avoid "SettingWithCopyWarning" from pandas
X, y = bikes.data.copy(), bikes.target

# We use only a subset of the data to speed up the example.
X = X.iloc[::5, :]
y = y[::5]

# %%
# The feature `"weather"` has a particularity: the category `"heavy_rain"` is a rare
# category.
X["weather"].value_counts()

# %%
# Because of this rare category, we collapse it into `"rain"`.
X["weather"] = (
    X["weather"]
    .astype(object)
    .replace(to_replace="heavy_rain", value="rain")
    .astype("category")
)

# %%
# We now have a closer look at the `"year"` feature:
X["year"].value_counts()

# %%
# We see that we have data from two years. We use the first year to train the
# model and the second year to test the model.
mask_training = X["year"] == 0.0
X = X.drop(columns=["year"])
X_train, y_train = X[mask_training], y[mask_training]
X_test, y_test = X[~mask_training], y[~mask_training]

# %%
# We can check the dataset information to see that we have heterogeneous data types. We
# have to preprocess the different columns accordingly.
X_train.info()

# %%
# From the previous information, we will consider the `category` columns as nominal
# categorical features. In addition, we will consider the date and time information as
# categorical features as well.
#
# We manually define the columns containing numerical and categorical
# features.
numerical_features = [
    "temp",
    "feel_temp",
    "humidity",
    "windspeed",
]
categorical_features = X_train.columns.drop(numerical_features)

# %%
# Before we go into the details regarding  the different machine
# learning pipelines, we will try to get some additional intuition regarding the dataset
# that will be helpful to understand the model's statistical performance and results of
# the partial dependence analysis.
#
# We plot the average number of bike rentals by grouping the data by season and
# by year.
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

days = ("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")
hours = tuple(range(24))
xticklabels = [f"{day}\n{hour}:00" for day, hour in product(days, hours)]
xtick_start, xtick_period = 6, 12

fig, axs = plt.subplots(nrows=2, figsize=(8, 6), sharey=True, sharex=True)
average_bike_rentals = bikes.frame.groupby(
    ["year", "season", "weekday", "hour"], observed=True
).mean(numeric_only=True)["count"]
for ax, (idx, df) in zip(axs, average_bike_rentals.groupby("year")):
    df.groupby("season", observed=True).plot(ax=ax, legend=True)

    # decorate the plot
    ax.set_xticks(
        np.linspace(
            start=xtick_start,
            stop=len(xticklabels),
            num=len(xticklabels) // xtick_period,
        )
    )
    ax.set_xticklabels(xticklabels[xtick_start::xtick_period])
    ax.set_xlabel("")
    ax.set_ylabel("Average number of bike rentals")
    ax.set_title(
        f"Bike rental for {'2010 (train set)' if idx == 0.0 else '2011 (test set)'}"
    )
    ax.set_ylim(0, 1_000)
    ax.set_xlim(0, len(xticklabels))
    ax.legend(loc=2)

# %%
# The first striking difference between the train and test set is that the number of
# bike rentals is higher in the test set. For this reason, it will not be surprising to
# get a machine learning model that underestimates the number of bike rentals. We
# also observe that the number of bike rentals is lower during the spring season. In
# addition, we see that during working days, there is a specific pattern around 6-7
# am and 5-6 pm with some peaks of bike rentals. We can keep in mind these different
# insights and use them to understand the partial dependence plot.
#
# Preprocessor for machine-learning models
# ----------------------------------------
#
# Since we later use two different models, a
# :class:`~sklearn.neural_network.MLPRegressor` and a
# :class:`~sklearn.ensemble.HistGradientBoostingRegressor`, we create two different
# preprocessors, specific for each model.
#
# Preprocessor for the neural network model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We will use a :class:`~sklearn.preprocessing.QuantileTransformer` to scale the
# numerical features and encode the categorical features with a
# :class:`~sklearn.preprocessing.OneHotEncoder`.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

mlp_preprocessor = ColumnTransformer(
    transformers=[
        ("num", QuantileTransformer(n_quantiles=100), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)
mlp_preprocessor

# %%
# Preprocessor for the gradient boosting model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# For the gradient boosting model, we leave the numerical features as-is and only
# encode the categorical features using a
# :class:`~sklearn.preprocessing.OrdinalEncoder`.
from sklearn.preprocessing import OrdinalEncoder

hgbdt_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(), categorical_features),
        ("num", "passthrough", numerical_features),
    ],
    sparse_threshold=1,
    verbose_feature_names_out=False,
).set_output(transform="pandas")
hgbdt_preprocessor

# %%
# 1-way partial dependence with different models
# ----------------------------------------------
#
# In this section, we will compute 1-way partial dependence with two different
# machine-learning models: (i) a multi-layer perceptron and (ii) a
# gradient-boosting model. With these two models, we illustrate how to compute and
# interpret partial dependence plots (PDP) for both numerical and categorical
# features and individual conditional expectation (ICE).
#
# Multi-layer perceptron
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Let's fit a :class:`~sklearn.neural_network.MLPRegressor` and compute
# single-variable partial dependence plots.
from time import time

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

print("Training MLPRegressor...")
tic = time()
mlp_model = make_pipeline(
    mlp_preprocessor,
    MLPRegressor(
        hidden_layer_sizes=(30, 15),
        learning_rate_init=0.01,
        early_stopping=True,
        random_state=0,
    ),
)
mlp_model.fit(X_train, y_train)
print(f"done in {time() - tic:.3f}s")
print(f"Test R2 score: {mlp_model.score(X_test, y_test):.2f}")

# %%
# We configured a pipeline using the preprocessor that we created specifically for the
# neural network and tuned the neural network size and learning rate to get a reasonable
# compromise between training time and predictive performance on a test set.
#
# Importantly, this tabular dataset has very different dynamic ranges for its
# features. Neural networks tend to be very sensitive to features with varying
# scales and forgetting to preprocess the numeric feature would lead to a very
# poor model.
#
# It would be possible to get even higher predictive performance with a larger
# neural network but the training would also be significantly more expensive.
#
# Note that it is important to check that the model is accurate enough on a
# test set before plotting the partial dependence since there would be little
# use in explaining the impact of a given feature on the prediction function of
# a model with poor predictive performance. In this regard, our MLP model works
# reasonably well.
#
# We will plot the averaged partial dependence.
import matplotlib.pyplot as plt

from hidimstat.marginal.partial_dependence_plot import PartialDependencePlot

common_params = {
    "n_jobs": 2,
    "grid_resolution": 20,
}

print("Computing partial dependence plots...")
features_info = {
    # features of interest
    "features": ["temp", "humidity", "windspeed", "season", "weather", "hour"],
    # information regarding categorical features
    "categorical_features": categorical_features,
}
tic = time()
pdp = PartialDependencePlot(
    mlp_model,
    **features_info,
    **common_params,
)
pdp.fit_importance(X=X_train)
# plot the figure
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(9, 8), constrained_layout=True)
for i, ax in enumerate(axs.ravel()):
    _ = pdp.plot(feature_id=i, ax=ax, X=X_train)

print(f"done in {time() - tic:.3f}s")
_ = fig.suptitle(
    (
        "Partial dependence of the number of bike rentals\n"
        "for the bike rental dataset with an MLPRegressor"
    ),
    fontsize=16,
)

# %%
# Gradient boosting
# ~~~~~~~~~~~~~~~~~
#
# Let's now fit a :class:`~sklearn.ensemble.HistGradientBoostingRegressor` and
# compute the partial dependence on the same features. We also use the
# specific preprocessor we created for this model.
from sklearn.ensemble import HistGradientBoostingRegressor

print("Training HistGradientBoostingRegressor...")
tic = time()
hgbdt_model = make_pipeline(
    hgbdt_preprocessor,
    HistGradientBoostingRegressor(
        categorical_features=categorical_features,
        random_state=0,
        max_iter=50,
    ),
)
hgbdt_model.fit(X_train, y_train)
print(f"done in {time() - tic:.3f}s")
print(f"Test R2 score: {hgbdt_model.score(X_test, y_test):.2f}")

# %%
# Here, we used the default hyperparameters for the gradient boosting model
# without any preprocessing as tree-based models are naturally robust to
# monotonic transformations of numerical features.
#
# Note that on this tabular dataset, Gradient Boosting Machines are both
# significantly faster to train and more accurate than neural networks. It is
# also significantly cheaper to tune their hyperparameters (the defaults tend
# to work well while this is not often the case for neural networks).
#
# We will plot the partial dependence for some of the numerical and categorical
# features.
print("Computing partial dependence plots...")
tic = time()
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(9, 8), constrained_layout=True)
pdp = PartialDependencePlot(
    hgbdt_model,
    **features_info,
    **common_params,
)
pdp.fit_importance(X=X_train)
for i, ax in enumerate(axs.ravel()):
    _ = pdp.plot(feature_id=i, ax=ax, X=X_train)
print(f"done in {time() - tic:.3f}s")
_ = fig.suptitle(
    (
        "Partial dependence of the number of bike rentals\n"
        "for the bike rental dataset with a gradient boosting"
    ),
    fontsize=16,
)

# %%
# Analysis of the plots
# ~~~~~~~~~~~~~~~~~~~~~
#
# We will first look at the PDPs for the numerical features. For both models, the
# general trend of the PDP of the temperature is that the number of bike rentals is
# increasing with temperature. We can make a similar analysis but with the opposite
# trend for the humidity features. The number of bike rentals is decreasing when the
# humidity increases. Finally, we see the same trend for the wind speed feature. The
# number of bike rentals is decreasing when the wind speed is increasing for both
# models. We also observe that :class:`~sklearn.neural_network.MLPRegressor` has much
# smoother predictions than :class:`~sklearn.ensemble.HistGradientBoostingRegressor`.
#
# Now, we will look at the partial dependence plots for the categorical features.
#
# We observe that the spring season is the lowest bar for the season feature. With the
# weather feature, the rain category is the lowest bar. Regarding the hour feature,
# we see two peaks around the 7 am and 6 pm. These findings are in line with the
# the observations we made earlier on the dataset.
#
# However, it is worth noting that we are creating potential meaningless
# synthetic samples if features are correlated.
#
# .. _ice-vs-pdp:
#
# ICE vs. PDP
# ~~~~~~~~~~~
#
# PDP is an average of the marginal effects of the features. We are averaging the
# response of all samples of the provided set. Thus, some effects could be hidden. In
# this regard, it is possible to plot each individual response. This representation is
# called the Individual Effect Plot (ICE). In the plot below, we plot 50 randomly
# selected ICEs for the temperature and humidity features.
print("Computing partial dependence plots and individual conditional expectation...")
tic = time()
# sphinx_gallery_thumbnail_number = 4
fig, axs = plt.subplots(ncols=2, figsize=(6, 4), sharey=True, constrained_layout=True)

features_info = {
    "features": ["temp", "humidity"],
    # "centered": True,
}

pdp = PartialDependencePlot(
    hgbdt_model,
    **features_info,
    **common_params,
)
# plot the figure
pdp.fit_importance(X=X_train)
for i, ax in enumerate(axs.ravel()):
    _ = pdp.plot(feature_id=i, ax=ax, X=X_train)
print(f"done in {time() - tic:.3f}s")
_ = fig.suptitle("ICE and PDP representations", fontsize=16)

# %%
# We see that the ICE for the temperature feature gives us some additional information:
# Some of the ICE lines are flat while some others show a decrease of the dependence
# for temperature above 35 degrees Celsius. We observe a similar pattern for the
# humidity feature: some of the ICEs lines show a sharp decrease when the humidity is
# above 80%.
#
# Not all ICE lines are parallel, this indicates that the model finds
# interactions between features. We can repeat the experiment by constraining the
# gradient boosting model to not use any interactions between features using the
# parameter `interaction_cst`:
from sklearn.base import clone

interaction_cst = [[i] for i in range(X_train.shape[1])]
hgbdt_model_without_interactions = (
    clone(hgbdt_model)
    .set_params(histgradientboostingregressor__interaction_cst=interaction_cst)
    .fit(X_train, y_train)
)
print(f"Test R2 score: {hgbdt_model_without_interactions.score(X_test, y_test):.2f}")

# %%
fig, axs = plt.subplots(ncols=2, figsize=(6, 4), sharey=True, constrained_layout=True)

# features_info["centered"] = False
pdp = PartialDependencePlot(
    hgbdt_model_without_interactions,
    **features_info,
    **common_params,
)
pdp.fit_importance(X=X_train)
for i, ax in enumerate(axs.ravel()):
    _ = pdp.plot(feature_id=i, ax=ax, X=X_train)
_ = fig.suptitle("ICE and PDP representations", fontsize=16)

# %%
# .. _plt_partial_dependence_custom_values:
#
# Custom Inspection Points
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# None of the examples so far specify _which_ points are evaluated to create the
# partial dependence plots. By default we use percentiles defined by the input dataset.
# In some cases it can be helpful to specify the exact points where one would like the
# model evaluated. For instance, if a user wants to test the model behavior on
# out-of-distribution data or compare two models that were fit on slightly different
# data. The `custom_values` parameter allows the user to pass in the values that they
# want the model to be evaluated on. This overrides the `grid_resolution` and
# `percentiles` parameters. Let's return to our gradient boosting example above
# but with custom values

print("Computing partial dependence plots with custom evaluation values...")
tic = time()
fig, axs = plt.subplots(ncols=2, figsize=(6, 4), sharey=True, constrained_layout=True)

features_info = {
    "features": ["temp", "humidity"],
}

pdp = PartialDependencePlot(
    hgbdt_model,
    **features_info,
    **common_params,
    # we set custom values for temp feature -
    # all other features are evaluated based on the data
    custom_values={"temp": np.linspace(0, 40, 10)},
)
pdp.fit_importance(X=X_train)
for i, ax in enumerate(axs.ravel()):
    _ = pdp.plot(feature_id=i, ax=ax, X=X_train)
print(f"done in {time() - tic:.3f}s")
_ = fig.suptitle(
    (
        "Partial dependence of the number of bike rentals\n"
        "for the bike rental dataset with a gradient boosting"
    ),
    fontsize=16,
)
plt.show()
