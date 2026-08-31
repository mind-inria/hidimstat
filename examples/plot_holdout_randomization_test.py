"""
Feature selection with the Holdout Randomization Test
=====================================================

Here we show how to compute p-values with the **Holdout Randomization Test
(HRT)** [:footcite:t:`tansey2022holdout`]. We simulate a regression problem in
which only 10 of the 100 features carry signal, fit a linear model, and test
each feature with :class:`~hidimstat.CFI` by passing
``statistical_test="hrt"``. The resulting p-values are valid in finite samples
and make no assumption on the model; we check this on the features that carry
no signal.
"""

# %%
# Generating a synthetic dataset
# ------------------------------
# We simulate a regression dataset with ``make_regression``: out of the 100
# features, only 10 carry signal and the 90 others are pure noise. Simulated
# data gives us access to this ground truth, which is what lets us check the
# behavior of the test at the end of the example.

from sklearn.datasets import make_regression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

X, y, coef = make_regression(
    n_samples=1000,
    n_features=100,
    n_informative=10,
    noise=50,
    coef=True,
    random_state=0,
)
# The coefficients of the data-generating process give us the true support.
support = coef != 0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = RidgeCV()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"R² score on the test set: {r2_score(y_test, y_pred):.2f}")


# %%
# Computing the p-values
# ----------------------
# :class:`~hidimstat.CFI` measures the importance of a feature by the increase
# in risk caused by replacing it with draws from an estimate of its conditional
# distribution :math:`p(X_j \mid X_{-j})`, as in
# :ref:`sphx_glr_generated_gallery_examples_plot_cfi.py`. Those draws are
# exactly the ones the HRT needs. Write :math:`t_0` for the risk of the model
# on the original data, :math:`t_k` for its risk on the :math:`k`-th of the
# :math:`K` draws, and :math:`\Delta_k = t_k - t_0` for the difference between
# the two. The p-value of feature :math:`j` counts the draws that did not
# raise the risk:
#
# .. math::
#     p_j = \frac{1}{K + 1} \left( 1 + \sum_{k=1}^{K}
#     \mathbb{I}(\Delta_k \leq 0) \right)
#
# An important feature makes most draws worse, so most :math:`\Delta_k` are
# positive and :math:`p_j` is small. The :math:`+1` terms keep the p-value
# exact at finite :math:`K`.

from hidimstat import CFI

cfi = CFI(
    estimator=model,
    random_state=0,
    statistical_test="hrt",
    n_permutations=500,
    n_jobs=5,
)
cfi.fit(X_train, y_train)
cfi.importance(X_test, y_test)

p_values = cfi.pvalues_

# %%
# Looking at the p-values
# -----------------------
# We plot the p-value of every feature, separating the 10 informative features
# from the 90 null ones. The horizontal line is the usual level
# :math:`\alpha = 0.05`: a feature below it is declared important. The
# informative features sit at the smallest p-value the test can return,
# :math:`1 / (K + 1)`, while the null ones spread over the whole range, as
# they should when the null hypothesis holds.

import matplotlib.pyplot as plt
import numpy as np

alpha = 0.05
index = np.arange(len(p_values))

_, ax = plt.subplots(figsize=(6, 3.5))
ax.scatter(index[support], p_values[support], label="informative features")
ax.scatter(index[~support], p_values[~support], label="null features")
ax.axhline(alpha, ls="--", lw=1.5, color="black", label=rf"$\alpha$ = {alpha}")
ax.set_yscale("log")
ax.set_ylim(bottom=5e-4)
ax.set_xlabel("feature index")
ax.set_ylabel("p-value")
ax.legend(loc="lower center", ncols=3, fontsize="small")
plt.tight_layout()
plt.show()

# %%
# Visualizing the control of the type-I error
# -------------------------------------------
# Above we drew the line at :math:`\alpha = 0.05`, but that choice is
# arbitrary: the guarantee of the HRT holds at any level. For a feature that
# carries no information about ``y`` once the others are known,
# :math:`P(p \leq \alpha) \leq \alpha` for every :math:`\alpha`. In words,
# whatever threshold we pick, the fraction of null features we wrongly declare
# important is at most the threshold we asked for. So we sweep :math:`\alpha`
# from 0 to 1 and, at each value, count the fraction of null features that
# fall below it. If the p-values are calibrated, that fraction tracks
# :math:`\alpha` itself, i.e. the curve follows the diagonal; the guarantee
# asks that it never run above it.

# sphinx_gallery_thumbnail_number = 2

alphas = np.linspace(0, 1, 101)
null = p_values[~support][:, None]

_, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(alphas, np.mean(null <= alphas, axis=0), lw=2.5, label="null features")
ax.plot(
    [0, 1], [0, 1], ls="--", lw=1.5, color="black", label="perfect calibration"
)
ax.set_xlabel(r"nominal level $\alpha$")
ax.set_ylabel(r"fraction of null features with $p \leq \alpha$")
ax.legend()
plt.tight_layout()
plt.show()

# %%
# The curve tracks the diagonal closely, so the p-values of the null features
# are well calibrated. This does not, however, control the proportion of false
# discoveries among the selected features: that proportion also depends on how
# many features are informative, and keeping it in check needs a correction
# such as the Benjamini-Hochberg procedure available in ``fdr_threshold``.

# %%
# References
# ----------
# .. footbibliography::
