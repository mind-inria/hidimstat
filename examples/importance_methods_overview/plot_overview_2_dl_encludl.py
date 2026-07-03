"""
Desparsified Lasso and Ensemble Clustered Inference
===================================================
We present and illustrate the limitations of the Desparsified
Lasso (DL) method through a simulated 2D example where the number of features is higher than
the number of samples. We then present two methods that leverage the data's
spatial structure to build clusters and perform the inference at the cluster level,
with clustered DL (:class:`hidimstat.CluDL`) and with Ensembled CluDL (:class:`hidimstat.EnCluDL`).
"""

# %%
# Generating the data
# -------------------
# We begin by simulating 2D data where the support
# consists of four regions of neighboring pixels located in each corner of the 2D image.
# The target variable :math:`y` is a continuous variable generated from a linear model. To
# make the problem more challenging, the pixels are spatially correlated using a
# Gaussian filter.

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from hidimstat._utils.scenario import multivariate_simulation_spatial

n_samples = 100
shape = (40, 40)
n_features = shape[1] * shape[0]
roi_size = 5  # size of the edge of the four predictive regions
signal_noise_ratio = 16.0  # noise standard deviation
smooth_X = 1.0  # level of spatial smoothing introduced by the Gaussian filter

# generating the data
X_init, y, beta, epsilon = multivariate_simulation_spatial(
    n_samples, shape, roi_size, signal_noise_ratio, smooth_X, seed=0
)
print(
    f"Number of samples: {X_init.shape[0]}, Number of features: {n_features}"
)

# visualize the data (green: support, white: null)
cmap = ListedColormap(["white", "tab:green"])
fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"xticks": [], "yticks": []})
ax.imshow(
    beta.reshape(shape),
    cmap=cmap,
    vmin=0,
    vmax=1,
)

legend_handles = [
    Patch(facecolor="tab:green", edgecolor="k", label="Support"),
    Patch(facecolor="white", edgecolor="k", label="Null"),
]
ax.legend(handles=legend_handles, loc="lower center")
plt.tight_layout()


# %%
# Feature importance with Desparsified Lasso
# ------------------------------------------
# Desparsified Lasso, also known as debiased Lasso, is a method that aims at estimating
# the regression coefficients. To do so, it uses coefficients obtained
# from a Lasso regression and corrects the bias induced by the L1-penalty. This method is
# particularly useful in high-dimensional settings where the number of features exceeds the number of samples.
# The feature importance thus corresponds to the estimated coefficients :math:`\hat{\beta}`.
# We perform inference using the Desparsified Lasso method, treating the data as
# a standard high-dimensional regression problem without considering its spatial
# structure. The aim of the inference step is to recover the support while controlling
# the Family-Wise Error Rate (FWER) at a targeted level of 0.1. To achieve this, we use
# the Bonferroni correction, applying a factor equal to the number of features. For more
# details about the Desparsified Lasso method, see :footcite:t:`javanmard2014confidence`,
# :footcite:t:`zhang2014confidence` and :footcite:t:`van2014asymptotically`.

import numpy as np
from sklearn.linear_model import LassoCV

from hidimstat import DesparsifiedLasso

fwer_target = 0.1
n_jobs = 5  # number of workers for parallel computing

# compute desparsified lasso
estimator = LassoCV(max_iter=1000, tol=0.0001, eps=0.01, fit_intercept=False)
dl = DesparsifiedLasso(estimator=estimator, n_jobs=n_jobs, random_state=0)
dl.fit_importance(X_init, y)

# compute estimated support, true positive, false positive, and false negative masks.
selected_dl = dl.fwer_selection(fwer=fwer_target, n_tests=n_features)


# %%
# We visualize the support estimated by the Desparsified Lasso method.


def visualize_selection(selected_features):
    tp_mask = ((selected_features.astype(int) == 1) & (beta == 1)).astype(bool)
    fp_mask = ((selected_features.astype(int) == 1) & (beta == 0)).astype(bool)
    fn_mask = ((selected_features.astype(int) == 0) & (beta == 1)).astype(bool)
    mask_dl = np.zeros(shape)
    mask_dl[tp_mask.reshape(shape)] = 1
    mask_dl[fp_mask.reshape(shape)] = -2
    mask_dl[fn_mask.reshape(shape)] = -1
    _, ax = plt.subplots(
        figsize=(4, 4), subplot_kw={"xticks": [], "yticks": []}
    )
    cmap = ListedColormap(["tab:red", "tab:purple", "white", "tab:green"])
    ax.imshow(mask_dl, cmap=cmap, vmin=-2, vmax=1)
    legend_handles = [
        Patch(facecolor="tab:green", edgecolor="k", label="True Positive"),
        Patch(facecolor="tab:red", edgecolor="k", label="False Positive"),
        Patch(facecolor="tab:purple", edgecolor="k", label="False Negative"),
    ]
    ax.legend(handles=legend_handles, loc="lower center")
    plt.tight_layout()


visualize_selection(selected_dl)


# %%
# We see that the Desparsified Lasso method is not powerful enough to recover the support,
# as it only selects scattered pixels as true positives without identifying the regions.
# We also see that the number of false positives is quite high, and sometimes false positives are
# located far from the true support. This is normal, as DL does not take into account the
# spatial structure of data.


# %%
# Clustered inference with CluDL
# -------------------------------
# To leverage the spatial structure of the data, we can group correlated pixels into clusters,
# using an additional spatial constraint (pixels are iteratively merged with neighboring pixels).
# This approach leads to a dimension reduction while preserving the data's spatial structure.
# To control the FWER at the targeted level of 0.1, we perform Bonferroni correction,
# but here the correction factor is equal to the number of clusters instead of the
# number of features. For more details about CluDL, see :footcite:t:`chevalier2022spatially`.

from sklearn.base import clone
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_extraction import image

from hidimstat import CluDL

# Clustering step
n_clusters = 200
connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1])
clustering = FeatureAgglomeration(
    n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
)

dl_2 = DesparsifiedLasso(estimator=clone(estimator), n_jobs=n_jobs)
clu_dl = CluDL(desparsified_lasso=dl_2, clustering=clustering, random_state=0)
clu_dl.fit_importance(X_init, y)
selected_cdl = clu_dl.fwer_selection(fwer=fwer_target, n_tests=n_clusters)

visualize_selection(selected_cdl)


# %%
# With ``CluDL``, the support recovery is more powerful, with large regions of the true
# support being correctly identified. However, some false positives remain. In this
# case, these false positives consist of small clusters that can be either contiguous to
# the true support or located far from it.


# %%
# Inference with Ensembled CluDL
# ------------------------------
# Finally, we perform inference using an ensembled version of ``CluDL`` called ``EnCluDL``. The
# idea is to run several ``CluDL`` algorithms with different clustering choices. By
# repeating the clustering step on different bootstrap samples of the data, this
# approach derandomizes the clustering choice and makes it more robust to small
# variations in the data. It can be efficiently parallelized since the different ``CluDL``
# runs are independent. Similar to ``CluDL``, we perform Bonferroni correction with a factor
# equal to the number of clusters on the p-values obtained by aggregating the different ``CluDL`` runs.
# For more details about ``EnCluDL``, see :footcite:t:`chevalier2022spatially`.

from hidimstat.ensemble_clustered_inference import EnCluDL

enclu_dl = EnCluDL(
    desparsified_lasso=DesparsifiedLasso(estimator=clone(estimator), n_jobs=1),
    clustering=clustering,
    n_bootstraps=20,
    random_state=0,
    n_jobs=n_jobs,
    cluster_bootstrap_size=0.5,
)
enclu_dl.fit_importance(X_init, y)
selected_ecdl = enclu_dl.fwer_selection(fwer=fwer_target, n_tests=n_clusters)

visualize_selection(selected_ecdl)

# %%
# Compared to ``CluDL``, ``EnCluDL`` further improves the inference, primarily by reducing the
# number of false positives located far from the true support. An intuitive explanation
# for this improvement is that while false discoveries far from the true support may
# occur randomly in a single clustering, they are less likely to occur repeatedly in
# overlapping clusters obtained from different bootstrap samples of the data.


# %%
# Takeaways
# ---------
# Desparsified Lasso is a fast feature importance method, that gives an easy
# interpretation of its coefficients as feature importance. However, it assumes
# a linear relationship, which might not capture well real data structure.
# To mitigate this, a solution is to perform clustering on the data, and regroup features,
# by applying the ``CluDL`` method.
# We can further reduce the instability of clustering method, by repeating the
# clustering process on different bootstrap of the samples, as executed by
# `EnCluDL`.
#
# In practise, the choice of the number of clusters depends on several parameters, such as
# the structure of the data (a higher correlation between neighboring features
# enable a greater dimension reduction, i.e. a smaller number of clusters),
# the number of samples (small datasets require more dimension reduction) and
# the required spatial tolerance :footcite:t:`chevalier2022spatially` (small
# clusters lead to limited spatial uncertainty).


# %%
# References
# ----------
# .. footbibliography::
