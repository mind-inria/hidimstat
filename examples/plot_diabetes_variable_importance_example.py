"""
Variable Importance on diabetes dataset
=======================================

Variable Importance estimates the influence of a given input variable to the
prediction made by a model. To assess variable importance in a prediction
problem, :footcite:t:`breimanRandomForests2001` introduced the permutation
approach where the values are shuffled for one variable/column at a time. This
permutation breaks the relationship between the variable of interest and the
outcome. Following, the loss score is checked before and after this
substitution for any significant drop in the performance which reflects the
significance of this variable to predict the outcome. This ease-to-use solution
is demonstrated, in the work by
:footcite:t:`stroblConditionalVariableImportance2008`, to be affected by the
degree of correlation between the variables, thus biased towards truly
non-significant variables highly correlated with the significant ones and
creating fake significant variables. They introduced a solution for the Random
Forest estimator based on conditional sampling by performing sub-groups
permutation when bisecting the space using the conditioning variables of the
buiding process. However, this solution is exclusive to the Random Forest and 
is costly with high-dimensional settings.
:footcite:t:`Chamma_NeurIPS2023` introduced a new model-agnostic solution to
bypass the limitations of the permutation approach under the use of the
conditional schemes. The variable of interest does contain two types of
information: 1) the relationship with the remaining variables and 2) the
relationship with the outcome. The standard permutation, while breaking the
relationship with the outcome, is also destroying the dependency with the
remaining variables. Therefore, instead of directly permuting the variable of
interest, the variable of interest is predicted by the remaining
variables and the residuals of this prediction are permuted before
reconstructing the new version of the variable. This solution preserves the
dependency with the remaining variables.

In this example, we compare both the standard permutation and its conditional
variant approaches for variable importance on the diabetes dataset for the
single-level case. The aim is to see if integrating the new
statistically-controlled solution has an impact on the results.

References
----------
.. footbibliography::

"""

#############################################################################
# Imports needed for this script
# ------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn.base import clone
from sklearn.datasets import load_diabetes

from hidimstat.bbi import BlockBasedImportance
from hidimstat import compute_loco

plt.rcParams.update({"font.size": 14})

# Fixing the random seed
rng = np.random.RandomState(2024)

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Use or not a cross-validation with the provided learner
k_fold = 2
# Identifying the categorical (nominal, binary & ordinal) variables
variables_categories = {}

#############################################################################
# Measure the importance of variables using the CPI method
# ------------------------------

cpi_importance_list = []
for i, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {i}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    cpi = CPI(
        estimator=regressor_list[i],
        covariate_estimator=RidgeCV(alphas=np.logspace(-3, 3, 10)),
        # covariate_estimator=HistGradientBoostingRegressor(random_state=0,),
        n_perm=100,
        random_state=0,
        n_jobs=4,
    )
    cpi.fit(X_train, y_train)
    importance = cpi.predict(X_test, y_test)
    cpi_importance_list.append(importance)

#############################################################################
# Measure the importance of variables using the LOCO method
# ------------------------------

loco_importance_list = []

for i, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {i}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    loco = LOCO(
        estimator=regressor_list[i],
        random_state=0,
        n_jobs=4,
    )
    loco.fit(X_train, y_train)
    importance = loco.predict(X_test, y_test)
    loco_importance_list.append(importance)


#############################################################################
# Measure the importance of variables using the permutation method
# ------------------------------

pi_importance_list = []

for i, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {i}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    pi = PermutationImportance(
        estimator=regressor_list[i],
        n_perm=100,
        random_state=0,
        n_jobs=4,
    )
    pi.fit(X_train, y_train)
    importance = pi.predict(X_test, y_test)
    pi_importance_list.append(importance)


#############################################################################
# Plotting the comparison
# -----------------------

list_res = {"Permutation": [], "Conditional": [], "LOCO": []}
for index, _ in enumerate(diabetes.feature_names):
    list_res["Permutation"].append(pvals_permutation[index][0])
    list_res["Conditional"].append(pvals_conditional[index][0])
    list_res["LOCO"].append(pvals_loco[index])

x = np.arange(len(diabetes.feature_names))
width = 0.25  # the width of the bars
multiplier = 0
fig, ax = plt.subplots(figsize=(10, 10), layout="constrained")

for attribute, measurement in list_res.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1

ax.set_ylabel(r"$-log_{10}p_{val}$")
ax.set_xticks(x + width / 2, diabetes.feature_names)
ax.legend(loc="upper left", ncols=3)
ax.set_ylim(0, 3)
ax.axhline(y=-np.log10(0.05), color="r", linestyle="-")
plt.show()

#############################################################################
# Analysis of the results
# -----------------------
# While the standard permutation flags multiple variables to be significant for
# this prediction, the conditional permutation (the controlled alternative)
# shows an agreement for "bmi", "bp" and "s6" but also highlights the importance
# of "sex" in this prediction, thus reducing the input space to four significant
# variables. LOCO underlines the importance of one variable "bp" for this
# prediction problem.
#
