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
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from hidimstat.CPI import CPI
from hidimstat.LOCO import LOCO
from hidimstat.PermutationImportance import PermutationImportance

#############################################################################
# Load the diabetes dataset
# ------------------------------
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

#############################################################################
# Fit a baselien model on the diabetes dataset
# ------------------------------
# We use a Ridge regression model with a 10-fold cross-validation to fit the
# diabetes dataset.

n_folds = 10
regressor = RidgeCV(alphas=np.logspace(-3, 3, 10))
regressor_list = [clone(regressor) for _ in range(n_folds)]
kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
for i, (train_index, test_index) in enumerate(kf.split(X)):
    regressor_list[i].fit(X[train_index], y[train_index])
    score = r2_score(
        y_true=y[test_index],
        y_pred=regressor_list[i].predict(
            X[test_index]))
    mse = mean_squared_error(
        y_true=y[test_index],
        y_pred=regressor_list[i].predict(
            X[test_index]))

    print(f"Fold {i}: {score}")
    print(f"Fold {i}: {mse}")

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
# Define a function to compute the p-value from importance values
# ------------------------------
def compute_pval(vim):
    mean_vim = np.mean(vim, axis=0)
    std_vim = np.std(vim, axis=0)
    pval = norm.sf(mean_vim / std_vim)
    return np.clip(pval, 1e-10, 1 - 1e-10)

#############################################################################
# Analyze the results
# ------------------------------


cpi_vim_arr = np.array([x['importance'] for x in cpi_importance_list]) / 2 
cpi_pval = compute_pval(cpi_vim_arr)

vim = [
    pd.DataFrame({
        'var': np.arange(cpi_vim_arr.shape[1]),
        'importance': x['importance'],
        'fold': i,
        'pval': cpi_pval,
        'method': 'CPI',

    }) for x in cpi_importance_list]

loco_vim_arr = np.array([x['importance'] for x in loco_importance_list])
loco_pval = compute_pval(loco_vim_arr)

vim += [
    pd.DataFrame({
        'var': np.arange(loco_vim_arr.shape[1]),
        'importance': x['importance'],
        'fold': i,
        'pval': loco_pval,
        'method': 'LOCO',
    }) for x in loco_importance_list]

pi_vim_arr = np.array([x['importance'] for x in pi_importance_list])
pi_pval = compute_pval(pi_vim_arr)

vim += [
    pd.DataFrame({
        'var': np.arange(pi_vim_arr.shape[1]),
        'importance': x['importance'],
        'fold': i,
        'pval': pi_pval,
        'method': 'PI',
    }) for x in pi_importance_list]

fig, ax = plt.subplots()

df_plot = pd.concat(vim)
df_plot['pval'] = -np.log10(df_plot['pval'])
im = sns.barplot(
    data=df_plot,
    x='var',
    y='pval',
    hue='method',
    ax=ax,
    palette='muted',
    dodge=True,
)
ax.set_ylabel(r'$-\log_{10}(\text{p-value})$')
ax.axhline(-np.log10(0.05), color='tab:red', ls='--')
ax.set_xlabel('Variable')
ax.set_xticklabels(diabetes.feature_names)

sns.despine(ax=ax)
plt.show()