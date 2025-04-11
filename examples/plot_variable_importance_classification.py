"""
Variable Importance for Classification
=============================================================
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import ttest_1samp
from sklearn.base import clone
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import balanced_accuracy_score, hinge_loss, log_loss
from sklearn.model_selection import KFold
from sklearn.svm import SVC

from hidimstat import CPI, LOCO, PermutationImportance, desparsified_lasso

# %%
#############################################################################
# Generate data where classes are not linearly separable
# --------------------------------------------------------------
rng = np.random.RandomState(0)
dataset = load_wine()
X, y = dataset.data, dataset.target
labels = dataset.feature_names
labels
# %%
cv = KFold(n_splits=5, shuffle=True, random_state=0)

df_list = []
model_linear = LogisticRegressionCV(Cs=np.logspace(-3, 3, 10))
model_svc = SVC(kernel="rbf", random_state=0)


def run_one_fold(X, y, model, train_index, test_index):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model_ = clone(model)
    model_.fit(X_train, y_train)
    y_pred = model_.predict(X_test)

    return balanced_accuracy_score(y_test, y_pred)


scores = Parallel(n_jobs=5)(
    delayed(run_one_fold)(X, y, model_linear, train_index, test_index)
    for train_index, test_index in cv.split(X)
)
scores
# %%


for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model_linear_ = clone(model_linear)
    model_linear_.fit(X_train, y_train)
    model_svc_ = clone(model_svc)
    model_svc_.fit(X_train, y_train)

    cpi_linear = CPI(
        estimator=model_linear_,
        imputation_model_continuous=RidgeCV(alphas=np.logspace(-3, 3, 10)),
        n_permutations=50,
        random_state=rng,
        method="predict_proba",
        loss=log_loss,
    )
    cpi_linear.fit(X_train, y_train)
    df_list.append(
        pd.DataFrame(
            {
                "importance": cpi_linear.importance(X_test, y_test)["importance"],
                "feature": labels,
                "method": "CPI",
                "model": "linear",
            }
        )
    )

    # Compute importance with LOCO
    loco_linear = LOCO(
        estimator=model_linear_,
        loss=log_loss,
        method="predict_proba",
    )
    loco_linear.fit(X_train, y_train)
    df_list.append(
        pd.DataFrame(
            {
                "importance": loco_linear.importance(X_test, y_test)["importance"],
                "feature": labels,
                "method": "LOCO",
                "model": "linear",
            }
        )
    )

    pfi_linear = PermutationImportance(
        estimator=model_linear_,
        n_permutations=50,
        random_state=rng,
        method="predict_proba",
        loss=log_loss,
    )
    pfi_linear.fit(X_train, y_train)
    df_list.append(
        pd.DataFrame(
            {
                "importance": pfi_linear.importance(X_test, y_test)["importance"],
                "feature": labels,
                "method": "Permutation",
                "model": "linear",
            }
        )
    )


# %%
df = pd.concat(
    df_list,
)
fig, ax = plt.subplots()

df_sub = df.query("model == 'linear'")
sns.boxplot(
    data=df_sub, x="importance", y="feature", hue="method", palette="muted", ax=ax
)

# %%
# Compute the correlation matrix
correlation_matrix = np.corrcoef(X.T)

# Plot the lower triangle of the correlation matrix
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    correlation_matrix,
    mask=mask,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    cbar=True,
    square=True,
    ax=ax,
)
ax.set_title("Correlation Matrix (Lower Triangle)")
plt.show()
# %%
