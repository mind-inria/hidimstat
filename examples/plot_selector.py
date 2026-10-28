"""
Feature selection pipeline with Conditional Feature Importance (CFI) on the wine dataset
========================================================================================

This example demonstrates how to perform feature selection from one of Hidimstat's selectors,
through feature importance measures using CFI [:footcite:t:`Chamma_NeurIPS2023`] on the wine dataset.
The data are the results of chemical analyses of wines grown in the same region in Italy,
derived from three different cultivars. Thirteen features are used to predict three types
of wine, making this a 3-class classification problem.

In this example, we show how to perform selection from p-values as standalone, and how to
include our selectors in a standard sklearn-compatible pipeline.
"""

# %%
# Standalone feature selection from p-values
# ------------------------------------------
# We start by loading the dataset, and show how to perform a basic
# feature selection with a CFI estimator based on a Multi-Layer Perceptron (MLP)
# classifier.
#
# As the Selector is designed to be integrated in a standard sklearn pipeline,
# a cross-validation operation is executed internally for Hidimstat's perturbed-based
# methods, to correctly estimate feature importance values. For this reason, Hidimstat
# Selectors expect the cross-validation version of a perturbation-based estimator, such
# as Permutation Feature Importance (PFI), CFI, Leave-One-Covariate-In and Out (LOCI / LOCO).

import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hidimstat import CFICV
from hidimstat.feature_selection import PValueSelect

# Load dataset
X, y = load_wine(return_X_y=True)
feature_names = np.array(load_wine().feature_names)

# We create noisy duplicates to test the selection process:
rng = np.random.default_rng(0)
features_to_duplicate = [
    "flavanoids",
    "proline",
    "color_intensity",
    "alcohol",
    "hue",
]
noise_level = 0.5

idx = [list(feature_names).index(name) for name in features_to_duplicate]
noise = rng.normal(
    loc=0.0,
    scale=noise_level * X[:, idx].std(axis=0),
    size=(X.shape[0], len(idx)),
)
X_replicates = X[:, idx] + noise

X = np.hstack([X, X_replicates])
feature_names = np.concatenate(
    [feature_names, [f"{name}_noisy_copy" for name in features_to_duplicate]]
)

# %%
# Now we define and fit our classifier as well as the Hidimstat feature importance
# estimator and the Selector from the importance p-values with a standard
# threshold of :math:`p<0.05`.

clf = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(100),
        random_state=0,
        max_iter=500,
    ),
)

cficv = CFICV(
    estimators=clf,
    cv=StratifiedKFold(),
    scoring="log_loss",
    imputation_model_continuous=RidgeCV(),
    feature_groups={
        feat_name: [i] for i, feat_name in enumerate(feature_names)
    },
    random_state=0,
)

selector = PValueSelect(estimator=cficv, threshold_max=0.1)
X_transformed = selector.fit_transform(X, y)

# %%
# We can now inspect how many features were selected, and which ones:

print(
    f"Initial number of features: {X.shape[1]}. Number of selected features: {X_transformed.shape[1]}"
)
print(f"Selected feature names: {feature_names[selector.selected_]}")

# %%
# Integrating PValueSelect in a pipeline
# --------------------------------------
# We show here how to directly integrate the feature selection process
# in a pipeline. The main difference here is that the StandardScaler
# is not needed in a pipeline for the CFICV estimator as it is directly
# instantiated in the global pipeline before the feature selection process.
# The pipeline is therefore constructed as follows:

import matplotlib.pyplot as plt

mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(100),
    random_state=0,
    max_iter=500,
)

pipeline = make_pipeline(
    StandardScaler(),
    PValueSelect(
        estimator=CFICV(
            estimators=mlp_classifier,
            cv=StratifiedKFold(),
            scoring="log_loss",
            imputation_model_continuous=RidgeCV(),
            feature_groups={
                feat_name: [i] for i, feat_name in enumerate(feature_names)
            },
            random_state=0,
        ),
        threshold_max=0.1,
    ),
    mlp_classifier,
)

initial_scores = []
scores = []

cv = StratifiedKFold(n_splits=5)

for train_idx, test_idx in cv.split(X, y):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Compute score from pipeline with feature selection
    pipeline.fit(X_train, y_train)
    scores.append(pipeline.score(X_test, y_test))

    # Compute score from initial pipeline without feature selection
    clf.fit(X_train, y_train)
    initial_scores.append(clf.score(X_test, y_test))

# %%
# Now that we have computed scores from the initial pipeline without feature selection,
# and scores with feature selection, we can plot the results with boxes to view
# their distributions.

_, ax = plt.subplots()
ax.boxplot(
    x=[initial_scores, scores],
    tick_labels=["Initial scores", "Scores with selection"],
)
ax.set_ylim(0.5, 1.1)
plt.show()
