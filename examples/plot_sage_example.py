"""
SAGE example
============
"""
# %%
# Regression SAGE example on the diabetes dataset
# -----------------------------------------------

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from hidimstat import SAGE

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = RidgeCV()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("R2 score:", r2_score(y_test, y_pred))

sage = SAGE(
    model,
    n_subsets=512,
    n_permutations=100,
    random_state=0,
    n_jobs=8,
)
sage.fit(X_train)
sage.importance(X_test, y_test)
ax = sage.plot_importance()


# %%
# Reproducing the example from the sage-values library
# ----------------------------------------------------

from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

bike_sharing = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True)
df = bike_sharing.frame
df = df[df["year"] == 0].drop(columns=["year"])

X = df.drop(columns=["count"]).to_numpy()
X = OrdinalEncoder().fit_transform(X)
y = df["count"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = RidgeCV()
model = HistGradientBoostingRegressor(random_state=0, max_depth=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("R2 score:", r2_score(y_test, y_pred))

sage = SAGE(
    model,
    n_subsets=512,
    n_permutations=10,
    random_state=0,
    n_jobs=8,
)
sage.fit(X_train)
subsample_size = 1024
rng = np.random.default_rng(0)
subsample_ids = rng.choice(len(X_test), size=subsample_size, replace=False)
sage.importance(X_test[subsample_ids], y_test[subsample_ids])
sage.plot_importance(feature_names=df.drop(columns=["count"]).columns.tolist())


# %%
# Classification SAGE example
# ---------------------------

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = MLPClassifier(random_state=0, hidden_layer_sizes=(256, 256))
print("Train set shape:", X_train.shape)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Balanced accuracy score:", balanced_accuracy_score(y_test, y_pred))

sage = SAGE(
    model,
    n_subsets=64,
    n_permutations=10,
    random_state=0,
    n_jobs=8,
    method="predict_proba",
    loss=log_loss,
)
sage.fit(X_train)
sage.importance(X_test, y_test)
ax = sage.plot_importance()
# %%
