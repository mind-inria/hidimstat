"""
Impact of the alpha term in d0CRT X-residual variance
======================================================

As discussed in `#118 <https://github.com/mind-inria/hidimstat/issues/118>`_,
the hidimstat implementation of d0CRT adds
``alpha * np.linalg.norm(coef_, ord=1)`` to the variance estimate of the
X-distillation residuals. No one really knows why, and this is not presented
in the paper.

This example compares the current implementation (with the alpha term) against
the paper's formula (without it), on two scenarios:

1. **Linear**: correlated Gaussian features, linear target (Toeplitz design).
2. **Non-linear**: correlated Gaussian features, Friedman-style target with
   HistGradientBoosting as the Y-estimator.

"""

# %%
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.linalg import toeplitz
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV
from tqdm import tqdm

from hidimstat import D0CRT
from hidimstat._utils.scenario import multivariate_simulation
from hidimstat.distilled_conditional_randomization_test import _joblib_distill
from hidimstat.statistical_tools.multiple_testing import fdp_power

# %%
# Patched distill function
# -----------------------------------------
# We override ``_joblib_distill`` to remove the alpha term, matching the
# formula from the paper: ``sigma2 = ||X_res||^2 / n``.


def _joblib_distill_no_alpha(
    idx,
    X,
    y,
    model_y,
    model_x,
    method,
    sigma_X=None,
    coefficient_minus_idx=None,
    is_logistic=False,
    intercept=None,
):
    """Same as _joblib_distill but sigma2 = mean(X_res^2), no alpha correction."""
    X_residual, sigma2, y_residual = _joblib_distill(
        idx,
        X,
        y,
        model_y,
        model_x,
        method,
        sigma_X,
        coefficient_minus_idx,
        is_logistic,
        intercept,
    )
    if sigma_X is None:
        n_samples = X.shape[0]
        X_minus_idx = np.delete(X, idx, 1)
        X_res = X[:, idx] - model_x.predict(X_minus_idx)
        sigma2 = np.linalg.norm(X_res) ** 2 / n_samples
    return X_residual, sigma2, y_residual


# %%
# Friedman non-linear scenario
# ---------------------------------
# Re-implementation of Friedman non-linear regression with correlated features.


def make_friedman_correlated(n_samples, n_features, rho, snr, seed=0):
    """Friedman-style target with correlated Gaussian features."""
    rng = np.random.default_rng(seed)
    p = max(n_features, 5)

    # Toeplitz covariance
    cov = toeplitz(rho ** np.arange(p))
    X = rng.multivariate_normal(np.zeros(p), cov, size=n_samples)

    # Friedman #1 target (5 active features)
    y_signal = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    )

    # Scale noise to match desired SNR
    sigma_noise = np.std(y_signal) / snr
    y = y_signal + rng.normal(0, sigma_noise, n_samples)

    beta = np.zeros(p)
    beta[:5] = 1.0  # indicator of active features

    return X, y, beta


# %%
# Run experiments
# ---------------

configs = [
    {"rho": 0.5, "snr": 1},
    {"rho": 0.5, "snr": 2},
    {"rho": 0.8, "snr": 1},
    {"rho": 0.8, "snr": 2},
]

n_samples = 200
n_features = 50
support_size = 5
n_repeats = 10
alpha = 0.1

results = []

for cfg in configs:
    for seed in tqdm(range(n_repeats)):
        # --- Linear scenario ---
        X_lin, y_lin, beta_lin, _ = multivariate_simulation(
            n_samples=n_samples,
            n_features=n_features,
            support_size=support_size,
            rho=cfg["rho"],
            signal_noise_ratio=cfg["snr"],
            seed=seed,
        )

        # --- Non-linear scenario ---
        X_nl, y_nl, beta_nl = make_friedman_correlated(
            n_samples=n_samples,
            n_features=n_features,
            rho=cfg["rho"],
            snr=cfg["snr"],
            seed=seed,
        )

        for scenario_name, X, y, beta, estimator in [
            (
                "Linear (Lasso)",
                X_lin,
                y_lin,
                beta_lin,
                LassoCV(random_state=seed),
            ),
            (
                "Non-linear (HGBR)",
                X_nl,
                y_nl,
                beta_nl,
                HistGradientBoostingRegressor(
                    random_state=seed, max_iter=30, max_depth=10
                ),
            ),
        ]:
            ground_truth = beta != 0

            for method_name, distill_fn in [
                ("with alpha", _joblib_distill),
                ("without alpha", _joblib_distill_no_alpha),
            ]:
                dcrt = D0CRT(
                    estimator=estimator,
                    screening_threshold=None,
                    random_state=seed,
                    n_jobs=6,
                )

                with mock.patch(
                    "hidimstat.distilled_conditional_randomization_test"
                    "._joblib_distill",
                    distill_fn,
                ):
                    dcrt.fit(X, y)
                    dcrt.importance(X, y)

                pvals = dcrt.pvalues_
                selected = pvals <= alpha
                fdp, power = fdp_power(selected, ground_truth)

                results.append(
                    {
                        "scenario": scenario_name,
                        "method": method_name,
                        "fdp": fdp,
                        "power": power,
                        "rho": cfg["rho"],
                        "snr": cfg["snr"],
                        "seed": seed,
                    }
                )

df = pd.DataFrame(results)

# %%
# Results visualization
# -------------------------

df_plot = df.copy()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))


sns.pointplot(
    data=df_plot,
    x="scenario",
    y="power",
    hue="method",
    ax=axes[0],
    capsize=0.1,
    ls="",
    dodge=0.3,
)
axes[0].set_title("power")

sns.pointplot(
    data=df_plot,
    x="scenario",
    y="fdp",
    hue="method",
    ax=axes[1],
    capsize=0.1,
    ls="",
    dodge=0.3,
)
axes[1].axhline(alpha, color="red", ls="--", label=f"FDR = {alpha}")
axes[1].legend()

sns.despine()
plt.tight_layout()
plt.show()

# %%
