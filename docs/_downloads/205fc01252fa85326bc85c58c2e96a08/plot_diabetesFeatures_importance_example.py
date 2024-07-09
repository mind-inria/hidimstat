"""
Variable Importance on diabetes dataset
=======================================

This example compares the standard permutation approach for variable importance
and its conditional variant on the diabetes dataset for the single-level case.
"""

#############################################################################
# Imports needed for this script
# ------------------------------

import numpy as np
from hidimstat.BBI import BlockBasedImportance
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})

# Fixing the random seed
rng = np.random.RandomState(2024)

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Use or not a cross-validation with the provided learner
k_fold = 2
# Identifying the categorical (nominal & ordinal) variables
list_nominal = {}

#############################################################################
# Standard Variable Importance
# ----------------------------

bbi_perm = BlockBasedImportance(
    estimator="RF",
    importance_estimator="Mod_RF",
    do_hyper=True,
    dict_hyper=None,
    conditional=False,
    group_stacking=False,
    prob_type="regression",
    k_fold=k_fold,
    list_nominal=list_nominal,
    n_jobs=10,
    verbose=0,
    n_perm=100,
)
bbi_perm.fit(X, y)
print("Computing the importance scores with standard permutation")
results_perm = bbi_perm.compute_importance()
pvals_perm = -np.log10(results_perm["pval"] + 1e-10)

#############################################################################
# Conditional Variable Importance
# -------------------------------

bbi_cond = BlockBasedImportance(
    estimator="RF",
    importance_estimator="Mod_RF",
    do_hyper=True,
    dict_hyper=None,
    conditional=True,
    group_stacking=False,
    prob_type="regression",
    k_fold=k_fold,
    list_nominal=list_nominal,
    n_jobs=10,
    verbose=0,
    n_perm=100,
)
bbi_cond.fit(X, y)
print("Computing the importance scores with conditional permutation")
results_cond = bbi_cond.compute_importance()
pvals_cond = -np.log10(results_cond["pval"] + 1e-5)

#############################################################################
# Plotting the comparison
# -----------------------

list_res = {"Perm": [], "Cond": []}
for ind_el, el in enumerate(diabetes.feature_names):
    list_res["Perm"].append(pvals_perm[ind_el][0])
    list_res["Cond"].append(pvals_cond[ind_el][0])

x = np.arange(len(diabetes.feature_names))
width = 0.25  # the width of the bars
multiplier = 0
fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")

for attribute, measurement in list_res.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1

ax.set_ylabel(r"$-log_{10}p_{val}$")
ax.set_xticks(x + width / 2, diabetes.feature_names)
ax.legend(loc="upper left", ncols=2)
ax.set_ylim(0, 3)
ax.axhline(y=-np.log10(0.05), color="r", linestyle="-")

plt.show()
