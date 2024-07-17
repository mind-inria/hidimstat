"""
Conditional sampling using residuals vs sampling Random Forest
==============================================================

"""

#############################################################################
# Imports needed for this script
# ------------------------------

from hidimstat import BlockBasedImportance
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import time

n, p = (1000, 12)
inter_cor, intra_cor = (0, 0.85)
n_blocks = 1
n_signal = 2
problem_type = "regression"
snr = 4
rf = RandomForestRegressor(random_state=2023)
dict_hyper = {"max_depth": [2, 5, 10, 20]}

#############################################################################
# Generate the synthetic data
# ---------------------------
# The function below generates the correlation matrix between the variables
# according to the provided degrees of correlation (intra + inter). (inter_cor)
# indicates the degree of correlation between the variables/groups whereas
# (intra_cor) specifies the corresponding degree between the variables within
# each group. For the single-level case, the (n_blocks) is set to 1 and the
# (intra_cor) illustrates the correlation between the variables.
#
# Next, we generate the synthetic data by randomly drawing n_signal predictors
# from the corresponding p variables and reordering the set of variables to put the
# n_signal predictors at the beginning. Following, the response is generated
# under a simple linear model with Gaussian noise.


def generate_cor_blocks(p, inter_cor, intra_cor, n_blocks):
    vars_per_grp = int(p / n_blocks)
    cor_mat = np.zeros((p, p))
    cor_mat.fill(inter_cor)
    for i in range(n_blocks):
        cor_mat[
            (i * vars_per_grp) : ((i + 1) * vars_per_grp),
            (i * vars_per_grp) : ((i + 1) * vars_per_grp),
        ] = intra_cor
    np.fill_diagonal(cor_mat, 1)
    return cor_mat


def _generate_data(seed):
    rng = np.random.RandomState(seed)

    cor_mat = generate_cor_blocks(p, inter_cor, intra_cor, n_blocks)
    x = norm.rvs(size=(p, n), random_state=seed)
    c = cholesky(cor_mat, lower=True)
    X = pd.DataFrame(np.dot(c, x).T, columns=[str(i) for i in np.arange(p)])

    data = X.copy()

    # Randomly draw n_signal predictors which are defined as signal predictors
    indices_var = list(rng.choice(range(data.shape[1]), size=n_signal, replace=False))

    # Reorder data matrix so that first n_signal predictors are the signal predictors
    # List of remaining indices
    indices_rem = [ind for ind in range(data.shape[1]) if ind not in indices_var]
    total_indices = indices_var + indices_rem
    # Before including the non-linear effects
    data = data.iloc[:, total_indices]
    data_signal = data.iloc[:, np.arange(n_signal)]

    # Determine beta coefficients
    effectset = [-0.5, -1, -2, -3, 0.5, 1, 2, 3]
    beta = rng.choice(effectset, size=data_signal.shape[1], replace=True)

    # Generate response
    # The product of the signal predictors with the beta coefficients
    prod_signal = np.dot(data_signal, beta)

    sigma_noise = np.linalg.norm(prod_signal, ord=2) / (
        snr * np.sqrt(data_signal.shape[0])
    )
    y = prod_signal + sigma_noise * rng.normal(size=prod_signal.shape[0])

    return data, y


#############################################################################
# Processing across multiple permutations
# ---------------------------------------
# In order to get statistical significance with p-values, we run the experiments
# across 100 repetitions.
#


def compute_simulations(seed):
    X, y = _generate_data(seed)
    # Using the residuals
    start_residuals = time.time()
    bbi_residual = BlockBasedImportance(
        estimator="DNN",
        importance_estimator="residuals_RF",
        do_hypertuning=True,
        dict_hypertuning=None,
        conditional=True,
        n_permutations=50,
        n_jobs=2,
        problem_type="regression",
        k_fold=2,
        variables_categories={},
    )
    bbi_residual.fit(X, y)
    results_bbi_residual = bbi_residual.compute_importance()

    df_residuals = {}
    df_residuals["method"] = ["residuals"] * X.shape[1]
    df_residuals["score"] = [results_bbi_residual["score_R2"]] * X.shape[1]
    df_residuals["elapsed"] = [time.time() - start_residuals] * X.shape[1]
    df_residuals["importance"] = np.ravel(results_bbi_residual["importance"])
    df_residuals["p-value"] = np.ravel(results_bbi_residual["pval"])
    df_residuals["iteration"] = [seed] * X.shape[1]
    df_residuals = pd.DataFrame(df_residuals)

    # Using the sampling RF
    start_sampling = time.time()
    bbi_sampling = BlockBasedImportance(
        estimator="DNN",
        importance_estimator="sampling_RF",
        do_hypertuning=True,
        dict_hypertuning=None,
        conditional=True,
        n_permutations=50,
        n_jobs=2,
        problem_type="regression",
        k_fold=2,
        variables_categories={},
    )
    bbi_sampling.fit(X, y)
    results_bbi_sampling = bbi_sampling.compute_importance()

    df_sampling = {}
    df_sampling["method"] = ["sampling"] * X.shape[1]
    df_sampling["score"] = [results_bbi_sampling["score_R2"]] * X.shape[1]
    df_sampling["elapsed"] = [time.time() - start_sampling] * X.shape[1]
    df_sampling["importance"] = np.ravel(results_bbi_sampling["importance"])
    df_sampling["p-value"] = np.ravel(results_bbi_sampling["pval"])
    df_sampling["iteration"] = [seed] * X.shape[1]
    df_sampling = pd.DataFrame(df_sampling)

    df_final = pd.concat([df_residuals, df_sampling], axis=0)
    return df_final


parallel = Parallel(n_jobs=2, verbose=1)
final_result = parallel(
    delayed(compute_simulations)(seed=seed) for seed in np.arange(1, 11)
)

#############################################################################
# Plotting AUC score and Type-I error
# -----------------------------------
# With the prediction problems turns to be a binary classification problem for
# the variables being relevant or non-relevant vs the ground-truth, we measure
# the performance in terms of type-I error i.e. the rate of true non-relevant
# variables detected as relevant and AUC score related to correct significant
# variables ordering.
#

df_final_result = pd.concat(final_result, axis=0).reset_index(drop=True)
df_auc = df_final_result.groupby(by=["method", "iteration"]).apply(
    lambda x: roc_auc_score([1] * n_signal + [0] * (p - n_signal), -x["p-value"])
)
df_auc = df_auc.reset_index(name="auc")
df_type_I = df_final_result.groupby(by=["method", "iteration"]).apply(
    lambda x: sum(x.iloc[n_signal:, :]["p-value"] <= 0.05) / x.iloc[2:, :].shape[0]
)
df_type_I = df_type_I.reset_index(name="type-I")

auc = [
    np.array(df_auc["auc"])[: int(df_auc.shape[0] / 2)],
    np.array(df_auc["auc"])[int(df_auc.shape[0] / 2) :],
]
typeI_error = [
    np.array(df_type_I["type-I"])[: int(df_type_I.shape[0] / 2)],
    np.array(df_type_I["type-I"])[int(df_type_I.shape[0] / 2) :],
]

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharey=True)

# AUC score
axs[0].violinplot(auc, showmeans=False, showmedians=True, vert=False)
axs[0].set_title("AUC score")
axs[0].xaxis.grid(True)
axs[0].set_yticks([x + 1 for x in range(len(auc))], labels=["Residuals", "Sampling"])
axs[0].set_ylabel("Method")

# Type-I Error
axs[1].violinplot(typeI_error, showmeans=False, showmedians=True, vert=False)
axs[1].set_title("Type-I Error")
axs[1].axvline(x=0.05, color="r", label="Nominal Rate")
plt.show()
