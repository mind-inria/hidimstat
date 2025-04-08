"""
Model agnostic variable importance when ANOVA fails
======================================================

In this example, we illustrate on the circles dataset the limitations of variable
importance methods based on linear models, such as the Analysis of Variance (ANOVA)
F-test. We then show how model agnostic methods such as Leave-One-Covariate-Out (LOCO)
allow to compute variable importance even when classes are not linearly separable.

To compute the importance of a variable, LOCO re-fits a sub-model on a subset of the
data where the variable of interest is removed. The importance of the variable is then
computed as the difference between the loss of the sub-model and the loss of the full
model. As shown in :footcite:t:`williamson2021nonparametric` this loss difference can
be interpreted as an unnormalized generalized ANOVA (difference of R2). Denoting
:math:`\mu` the predictive model used, :math:`\mu_{-j}` the sub-model where the j-th
variable is removed, and :math:`X^{-j}` the data with the j-th variable removed, the
loss difference can be expressed as:
.. math:: \psi_{j} = \mathbb{V}(y) \left[\left[ 1 - \frac{\mathbb{E}[(y - \mu(X))^2]}{\mathbb{V}(y)}\right] - \left[ 1 - \frac{\mathbb{E}[(y - \mu_{-j}(X^{-j}))^2]}{\mathbb{V}(y)}\right]\right]

where :math:`\psi_{j}` is the LOCO importance of the j-th variable.

References
----------
.. footbibliography::
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_1samp
from sklearn.base import clone
from sklearn.datasets import make_circles
from sklearn.feature_selection import f_classif
from sklearn.metrics import hinge_loss
from sklearn.model_selection import KFold
from sklearn.svm import SVC

from hidimstat import LOCO

#############################################################################
# Generate data where classes are not linearly separable
# --------------------------------------------------------------
rng = np.random.RandomState(0)
X, y = make_circles(n_samples=500, noise=0.1, factor=0.6, random_state=rng)

fig, ax = plt.subplots()
sns.scatterplot(
    x=X[:, 0],
    y=X[:, 1],
    hue=y,
    ax=ax,
    palette="muted",
)
ax.legend(title="Class")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
plt.show()


###############################################################################
# Compute variable importance using ANOVA and LOCO
# -----------------------------------------------------
# The ANOVA F-test is computed using the f_classif function from sklearn. The LOCO
# importance is computed using the LOCO class from hidimstat. We then compare the
# log10 p-values obtained from both methods and represent the commonly used 0.05
# threshold. The pvalues obtained with an ANOVA suggest that the two features are
# unimportant. However, the importances obtained by using LOCO with a SVC model
# reveal that the two features are in fact important.
_, pval_anova = f_classif(X, y)
print(f"ANOVA F-test pvalues: {pval_anova}")

cv = KFold(n_splits=5, shuffle=True, random_state=0)
model = SVC(kernel="rbf", random_state=0)

importances = []
for train, test in cv.split(X):
    model_c = clone(model)
    model_c.fit(X[train], y[train])

    vim = LOCO(estimator=model_c, loss=hinge_loss, method="decision_function", n_jobs=2)
    vim.fit(X[train], y[train])
    importances.append(vim.score(X[test], y[test])["importance"])

_, pval_loco = ttest_1samp(importances, 0, axis=0, alternative="greater")

df_pval = pd.DataFrame(
    {
        "pval": np.hstack([pval_anova, pval_loco]),
        "method": ["ANOVA"] * len(pval_anova) + ["LOCO"] * len(pval_loco),
        "Feature": ["X1", "X2"] * 2,
    }
)
df_pval["log10pval"] = -np.log10(df_pval["pval"])

fig, ax = plt.subplots()
sns.barplot(
    data=df_pval,
    y="Feature",
    x="log10pval",
    hue="method",
    palette="muted",
    ax=ax,
)
ax.set_xlabel("-$\log_{10}(pval)$")
ax.axvline(-np.log10(0.05), color="k", lw=3, linestyle="--", label="-$\log_{10}(0.05)$")
ax.legend()
