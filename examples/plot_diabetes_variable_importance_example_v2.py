# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn.base import clone
from sklearn.datasets import load_diabetes
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from hidimstat.CPI import CPI

# %%


def compute_pval(vim):
    mean_vim = np.mean(vim, axis=0)
    std_vim = np.std(vim, axis=0)
    pval = norm.sf(mean_vim / std_vim)
    return pval
# %%


diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
# %%

n_folds = 5
# regressor = HistGradientBoostingRegressor(random_state=0,)
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


# %%
importance_list = []
for i, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {i}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    cpi = CPI(
        estimator=regressor_list[i],
        # covariate_estimator=RidgeCV(alphas=np.logspace(-3, 3, 10)),
        covariate_estimator=HistGradientBoostingRegressor(random_state=0,),
        n_perm=100,
        groups=None,
        random_state=0,
    )
    cpi.fit(X_train, y_train)
    importance = cpi.predict(X_test, y_test)
    importance_list.append(importance)

# %%
vim_arr = np.array([x['importance'] for x in importance_list])
pval = compute_pval(vim_arr)

vim = [
    pd.DataFrame({
        'var': np.arange(10),
        'importance': x['importance'],
        'fold': i,
        'pval': pval,

    }) for x in importance_list]

# %%
fig, ax = plt.subplots()

im = sns.boxplot(
    data=pd.concat(vim),
    x='var',
    y='importance',
    hue='pval',
    ax=ax,
    palette='viridis_r',
    legend=False,
)
ax.set_xticklabels(diabetes.feature_names)


significant_vars = np.where(pval < 0.05)[0]
for var in significant_vars:
    ax.plot(var, vim_arr[:, var].mean(), '^', c='tab:red')

sns.despine(ax=ax)
