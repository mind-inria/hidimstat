"""
Tabular Foundation Model TabICL
================================

In this example, we demonstrate how to use a tabular foundation model such as
TabICL [:footcite:t:`qu2025tabicl`, :footcite:t:`qu2026tabiclv2`] with a
straightforward example on the California Housing regression dataset.
"""

# %%
# Loading data and TabICL
# -----------------------
# We start by loading data from the California Housing dataset and fitting the
# TabICL model. If this is the first use, the model will be downloaded and cached
# for future use. The main advantage is that the model does not require
# hyperparameter tuning as a pre-trained transformer model. We recommend switching
# to GPU through the adequate class parameter for datasets that go over 10k
# samples depending on the number of features due to computation time increase.

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tabicl import TabICLRegressor

dataset = fetch_california_housing()
X, y, feat_names = dataset.data, dataset.target, dataset.feature_names
X, y = resample(X, y, replace=False, n_samples=500, stratify=y, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = TabICLRegressor(device="cpu", kv_cache=True)
model.fit(X_train, y_train)
print(f"Model accuracy on the test data {model.score(X_test, y_test):.3}.")

# %%
# For the moment, TabICL can only be used with LOCI, LOCO, PFI, or CFI. TabICL only works with
# integer seeding, which is currently not supported by D0CRT.
#
# We use the "kv_cache" parameter for TabICL to cache column embedding key-value projections and row
# interaction outputs (representations) so that predictions are much faster since it computes the
# attention operations.
#
# TabICL interfaces with HiDimStat the same way as any other Scikit-learn estimator,
# which lets us compute feature importance as easily as follows:

import pandas as pd
import seaborn as sns

from hidimstat import CFI

cfi = CFI(
    estimator=model,
    scoring="mean_squared_error",
    n_permutations=5,
    n_jobs=-1,
)
cfi.fit(X_train, y_train)
importance = cfi.importance(X_test, y_test)
selection = cfi.fdr_selection(fdr=0.1)

# %%
# Let's wrap the results in a Pandas dataframe and plot the importance
# and the selected variables. We also display a second plot showing the
# predictability score of each group of variables $X^j$ from the others $X^{-j}$
# based on the conditional samplers fitted with CFI.

df = pd.DataFrame(
    {
        "feature": feat_names,
        "importance": importance,
        "selected": selection,
        "feature_predictability": cfi.feature_predictability_,
    }
).sort_values("importance", ascending=False)

import matplotlib.pyplot as plt

_, axes = plt.subplots(1, 2)

cfi.plot_importance(ax=axes[0])
axes[0].set_yticklabels(df["feature"].values)
axes[0].set_title("Feature importance")

sns.barplot(df, x="feature_predictability", y="feature", ax=axes[1])
sns.despine(ax=axes[1])
axes[1].set_ylabel("")
axes[1].set_yticks(ticks=[], labels=[])
axes[1].set_title("Feature predictability")

plt.tight_layout()
plt.show()

# %%
# As you can see from the timer under, even with 500 samples and 9 features,
# running TabICL on the CPU is slow. Using a GPU would allow for an important speedup.
