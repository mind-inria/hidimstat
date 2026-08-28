"""
Feature selection with the Holdout Randomization Test
=====================================================

Here we show how to compute p-values with the **Holdout Randomization Test
(HRT)** [:footcite:t:`tansey2022holdout`]. We simulate a regression problem in
which only 10 of the 100 features carry signal, fit a linear model, and test
each feature with :class:`~hidimstat.CFI` by passing ``statistical_test="hrt"``.
The resulting p-values are valid in finite samples and make no assumption on
the model; we check this on the features that carry no signal.
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
# exactly the ones the HRT needs. By default the loss differences they produce
# are summarized with a t-test; ``statistical_test="hrt"`` instead counts the
# draws that did *not* increase the risk, which is what removes the normality
# assumption. The model is never refitted, so each draw only costs one more
# prediction.

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
# Visualizing the control of the type-I error
# -------------------------------------------
# The HRT controls the type-I error of each null feature separately: for a
# feature that carries no information about ``y`` once the others are known,
# :math:`P(p \leq \alpha) \leq \alpha` at any level :math:`\alpha`. We plot
# that probability, measured across the null features, as a function of
# :math:`\alpha`. The diagonal is where the two are equal, and the guarantee
# asks that the curve not run above it.

import matplotlib.pyplot as plt
import numpy as np

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
