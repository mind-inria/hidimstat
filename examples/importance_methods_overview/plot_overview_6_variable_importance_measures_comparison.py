"""
Variable Importance Measures Comparison
=======================================
We have presented the coefficients of a linear model, Permutation Feature Importance (PFI),
Conditional Feature Importance (CFI), and Leave-One-Covariate-Out (LOCO). How can we relate
all of these quantities? We focus on a linear setting, where a linear model is sufficient
to understand the underlying data-generating process, and explore how we can compare different
methods.
"""

# %%
# Can we compare methods ?
# ------------------------
# One question that we may ask ourselves is the possibility of comparing different variable
# importance measures. Indeed, they each answer different questions, and seem to focus on
# different quantity measures.
# However, we can theoretically show that CFI is equal to twice LOCO. Therefore, we can rescale CFI
# by dividing it by 2 to match the LOCO scale. We proceed similarly for PFI, even though
# the underlying estimand is not necessarily the same. Is it measuring the same quantity
# in this setting?
# We start by defining two functions to generate linear data with an autoregressive structure
# with Toeplitz covariance matrix.

import numpy as np
from sklearn.model_selection import train_test_split

seed = 0
rng = np.random.default_rng(seed)


def random_covariance(p, strength=0.5):
    # random matrix
    A = rng.normal(size=(p, p))

    # make it symmetric PSD-like
    Sigma = A @ A.T

    # normalize to correlation matrix
    D = np.sqrt(np.diag(Sigma))
    Sigma = Sigma / np.outer(D, D)

    # shrink toward identity (controls correlation strength)
    Sigma = strength * Sigma + (1 - strength) * np.eye(p)

    return Sigma


def linear_data(n, p, correlation=0.25, sparsity=0.5):
    """
    Function to simulate linear data with an autoregressive structure with Toeplitz
    covariance matrix
    """
    # Number of non-null
    k = int(sparsity * p)

    # Generate the variables from a multivariate normal distribution
    mu = np.zeros(p)
    Sigma = np.array(
        [[(correlation) ** abs(i - j) for i in range(p)] for j in range(p)]
    )
    # covariance matrix of X
    Sigma = random_covariance(p, strength=correlation)
    X = rng.multivariate_normal(mu, Sigma, size=(n))
    # Generate the response from a linear model
    non_zero = rng.choice(p, k, replace=False)
    beta_true = np.zeros(p)
    beta_true[non_zero] = 1.0
    eps = rng.standard_normal(size=n)
    y = np.dot(X, beta_true) + eps

    return X, y, beta_true


X, y, beta = linear_data(1000, 10, correlation=0, sparsity=0.5)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=seed,
    shuffle=True,
)

# %%
# We now create the objects for the three different variable importance measures.

import pandas as pd
from sklearn.linear_model import LassoCV

from hidimstat import CFI, D0CRT, LOCO, PFI, ModelXKnockoff


def compute_importance(X_train, X_test, y_train, y_test):
    model = LassoCV()
    df_list = []

    model = model.fit(X_train, y_train)
    df_list.append(
        pd.DataFrame(
            {
                "VIM": "Coefficient",
                "feature": list(range(X.shape[1])),
                "importance": model.coef_**2,
                "model": model.__class__.__name__,
            }
        )
    )

    pfi = PFI(model)
    pfi.fit(X_train, y_train)
    importances = 0.5 * pfi.importance(X_test, y_test)
    df_list.append(
        pd.DataFrame(
            {
                "VIM": "PFI",
                "feature": list(range(X.shape[1])),
                "importance": importances,
                "model": model.__class__.__name__,
            }
        )
    )

    loco = LOCO(model)
    loco.fit(X_train, y_train)
    importances = loco.importance(X_test, y_test)
    df_list.append(
        pd.DataFrame(
            {
                "VIM": "LOCO",
                "feature": list(range(X.shape[1])),
                "importance": importances,
                "model": model.__class__.__name__,
            }
        )
    )
    cfi = CFI(model)
    cfi.fit(X_train, y_train)
    importances = 0.5 * cfi.importance(X_test, y_test)
    df_list.append(
        pd.DataFrame(
            {
                "VIM": "CFI",
                "feature": list(range(X.shape[1])),
                "importance": importances,
                "model": model.__class__.__name__,
            }
        )
    )
    return pd.concat(df_list)


df_plot = compute_importance(X_train, X_test, y_train, y_test)

# %%
# Now that we have computed importances with 3 methods, we plot and visually compare them :

import matplotlib.pyplot as plt
import seaborn as sns

ax = sns.barplot(
    data=df_plot,
    y="feature",
    x="importance",
    hue="VIM",
    palette="muted",
    orient="h",
)
sns.despine()

for i, support in enumerate(beta):
    if support != 0:
        ax.axhspan(
            i - 0.45,
            i + 0.45,
            color="tab:olive",
            alpha=0.3,
            zorder=-1,
            label="True Support" if i == 1 else None,
        )
ax.legend()
plt.show()

# %%
# We observe that all these Variable Importance Measures are very close and appear to target the
# same underlying quantity. Is this always the case? Why, in this setting, does Permutation
# Feature Importance (PFI) coincide with Conditional Feature Importance (CFI)?

# %%
# Increasing the Correlation
# --------------------------
# What happens if we increase the correlation between features? How does this affect Conditional
# Feature Importance (CFI), and how does it change the estimated coefficients?

X, y, beta = linear_data(10000, 10, correlation=0.6, sparsity=0.5)
df_plot = compute_importance(
    *train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )
)

ax = sns.barplot(
    data=df_plot,
    y="feature",
    x="importance",
    hue="VIM",
    palette="muted",
    orient="h",
)
sns.despine()

for i, support in enumerate(beta):
    if support != 0:
        ax.axhspan(
            i - 0.45,
            i + 0.45,
            color="tab:olive",
            alpha=0.3,
            zorder=-1,
            label="True Support" if i == 1 else None,
        )
ax.legend()
plt.show()

# %%
# We see here that CFI and LOCO are still in the same magnitude, but PFI values
# are higher than them. It is to be anticipated since features are now correlated
# and CFI properly takes into account this kind of feature property, while PFI
# overestimates these values.

# %%
# Comparison with other methods
# -----------------------------
# We compared LOCO, PFI, and CFI importance measures with previous examples,
# and showed the limits of comparing these methods depending on the correlations
# between variables.
# We can ask ourselves whether it makes sense to compare, for instance, CFI
# and Model-X Knockoffs (MXKO) ? While both methods assesses the conditional
# contribution of a variable, and thus often agree on the strongest predictors,
# they answer very different questions. Comparing their numerical importance
# values is not meaningful since they are defined on different scales and have
# different statistical interpretations. In this light, a reasonable comparison
# approach is to identify the variables selected by MXKO, and examine the overlap
# with the CFI variables ranking, and discuss disagreements. Since CFI doesn't select
# variables with False Discovery Rate (FDR) control, we implemented such method
# in the library to overcome this, and provide further statistical guarantees for
# feature selection, and provide a better ground to compare methods.
