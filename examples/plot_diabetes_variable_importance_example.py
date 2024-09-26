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
buiding process. However, this solution is exclusive to the Random Forest and is
costly with high-dimensional settings.
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
# Standard Variable Importance
# ----------------------------
# To apply the standard permutation, we use the implementation introduced by (Mi
# et al., Nature, 2021) where the significance is measured by the mean of
# -log10(p_value). For this example, the inference estimator is set to the
# Random Forest learner.
#

bbi_permutation = BlockBasedImportance(
    estimator="RF",
    importance_estimator="residuals_RF",
    do_hypertuning=True,
    dict_hypertuning=None,
    conditional=False,
    group_stacking=False,
    problem_type="regression",
    k_fold=k_fold,
    variables_categories=variables_categories,
    n_jobs=2,
    verbose=0,
    n_permutations=100,
)
bbi_permutation.fit(X, y)
print("Computing the importance scores with standard permutation")
results_permutation = bbi_permutation.compute_importance()
pvals_permutation = -np.log10(results_permutation["pval"] + 1e-10)

#############################################################################
# Conditional Variable Importance
# -------------------------------
# For the conditional permutation importance based on the two blocks (inference
# + importance), the estimators are set to the Random Forest learner. The
# significance is measured by the mean of -log10(p_value).
#

bbi_conditional = BlockBasedImportance(
    estimator="RF",
    importance_estimator="residuals_RF",
    do_hypertuning=True,
    dict_hypertuning=None,
    conditional=True,
    group_stacking=False,
    problem_type="regression",
    k_fold=k_fold,
    variables_categories=variables_categories,
    n_jobs=2,
    verbose=0,
    n_permutations=100,
)
bbi_conditional.fit(X, y)
print("Computing the importance scores with conditional permutation")
results_conditional = bbi_conditional.compute_importance()
pvals_conditional = -np.log10(results_conditional["pval"] + 1e-5)

#############################################################################
# Leave-One-Covariate-Out (LOCO)
# ------------------------------
# We compare the previous permutation-based approaches with a removal-based
# approach LOCO (Williamson et al., Journal of the American Statistical
# Association, 2021) where the variable of interest is removed and the inference
# estimator is retrained using the new features to compare the loss for any drop in the
# performance.
#

results_loco = compute_loco(X, y, use_dnn=False)
pvals_loco = -np.log10(results_loco["p_value"] + 1e-5)

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
