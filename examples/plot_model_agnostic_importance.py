r"""
Variable Selection Under Model Misspecification
=============================================================

In this example, we illustrate the limitations of variable selection methods based on
linear models using the circles dataset. We first use the distilled conditional
randomization test (d0CRT), which is based on linear models :footcite:t:`liu2022fast` and then
demonstrate how model-agnostic methods, such as Leave-One-Covariate-Out (LOCO), can
identify important variables even when classes are not linearly separable.

To evaluate the importance of a variable, LOCO re-fits a sub-model using a subset of the
data where the variable of interest is removed. The importance of the variable is
quantified as the difference in loss between the full model and the sub-model. As shown
in :footcite:t:`williamson_2021_nonparametric` , this loss difference can be interpreted as an
unnormalized generalized ANOVA (difference of R²).  Denoting :math:`\mu` the predictive
model used, :math:`\mu_{-j}` the sub-model where the j-th variable is removed, and
:math:`X^{-j}` the data with the j-th variable removed, the loss difference can be
expressed as:

.. math::
    \psi_{j} = \mathbb{V}(y) \left[ \left[ 1 - \frac{\mathbb{E}[(y - \mu(X))^2]}{\mathbb{V}(y)} \right] - \left[ 1 - \frac{\mathbb{E}[(y - \mu_{-j}(X^{-j}))^2]}{\mathbb{V}(y)} \right] \right]

where :math:`\psi_{j}` is the LOCO importance of the j-th variable.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_1samp
from sklearn.base import clone
from sklearn.datasets import make_circles
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import hinge_loss, log_loss
from sklearn.model_selection import KFold
from sklearn.svm import SVC

from hidimstat import D0CRT, LOCO

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
# Define a linear and a non-linear estimator
# ------------------------------------------
non_linear_model = SVC(kernel="rbf", random_state=0)
linear_model = LogisticRegressionCV(Cs=np.logspace(-3, 3, 5))

###############################################################################
# Compute p-values using d0CRT
# ---------------------------------------------------------------------------
# We first compute the p-values using d0CRT which performs a conditional independence
# test (:math:`H_0: X_j \perp\!\!\!\perp y | X_{-j}`) for each variable. However,
# this test is based on a linear model (LogisticRegression) and fails to reject the null
# in the presence of non-linear relationships.
d0crt_linear = D0CRT(estimator=clone(linear_model), screening_threshold=None)
d0crt_linear.fit_importance(X, y)
pval_dcrt_linear = d0crt_linear.pvalues_

d0crt_non_linear = D0CRT(estimator=clone(non_linear_model), screening_threshold=None)
d0crt_non_linear.fit_importance(X, y)
pval_dcrt_non_linear = d0crt_non_linear.pvalues_

################################################################################
# Compute p-values using LOCO
# ---------------------------------------------------------------------------
# We then compute the p-values using LOCO
# with a linear, and then a non-linear model. When using a
# misspecified model, such as a linear model for this dataset, LOCO fails to reject the null
# similarly to d0CRT. However, when using a non-linear model (SVC), LOCO is able to
# identify the important variables.
cv = KFold(n_splits=5, shuffle=True, random_state=0)

importances_linear = []
importances_non_linear = []
for train, test in cv.split(X):
    non_linear_model_ = clone(non_linear_model)
    linear_model_ = clone(linear_model)
    non_linear_model_.fit(X[train], y[train])
    linear_model_.fit(X[train], y[train])

    vim_linear = LOCO(
        estimator=linear_model_, loss=log_loss, method="predict_proba", n_jobs=2
    )
    vim_non_linear = LOCO(
        estimator=non_linear_model_,
        loss=hinge_loss,
        method="decision_function",
        n_jobs=2,
    )
    vim_linear.fit(X[train], y[train])
    vim_non_linear.fit(X[train], y[train])

    importances_linear.append(vim_linear.importance(X[test], y[test])["importance"])
    importances_non_linear.append(
        vim_non_linear.importance(X[test], y[test])["importance"]
    )


################################################################################
# To select variables using LOCO, we compute the p-values using a t-test over the
# importance scores.

_, pval_linear = ttest_1samp(importances_linear, 0, axis=0, alternative="greater")
_, pval_non_linear = ttest_1samp(
    importances_non_linear, 0, axis=0, alternative="greater"
)

df_pval = pd.DataFrame(
    {
        "pval": np.hstack(
            [pval_dcrt_linear, pval_dcrt_non_linear, pval_linear, pval_non_linear]
        ),
        "method": ["d0CRT-linear"] * 2
        + ["d0CRT-non-linear"] * 2
        + ["LOCO-linear"] * 2
        + ["LOCO-non-linear"] * 2,
        "Feature": ["X1", "X2"] * 4,
    }
)
df_pval["minus_log10_pval"] = -np.log10(df_pval["pval"])


#################################################################################
# Plot the :math:`-log_{10}(pval)` for each method and variable
# ---------------------------------------------------------------------------
fig, ax = plt.subplots()
sns.barplot(
    data=df_pval,
    y="Feature",
    x="minus_log10_pval",
    hue="method",
    palette="muted",
    ax=ax,
)
ax.set_xlabel("-$\\log_{10}(pval)$")
ax.axvline(
    -np.log10(0.05), color="k", lw=3, linestyle="--", label="-$\\log_{10}(0.05)$"
)
ax.legend()
plt.show()


#################################################################################
# As expected, when using linear models (d0CRT and LOCO-linear) that are misspecified,
# the varibles are not selected. This highlights the benefit of using model-agnostic
# methods such as LOCO, which allows for the use of models that are expressive enough
# to explain the data.


#################################################################################
# References
# ---------------------------------------------------------------------------
# .. footbibliography::
