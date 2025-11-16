"""
Ensemble Clustered Inference on 2D data
=======================================

In this example, we present how to perform inference on 2D simulated data. This setting
is particularly challenging since the number of features (pixels in 2D) is much larger
than the number of samples. We first illustrates the limitations of the Desparsified
Lasso method in this setting and then present two methods that leverage the spatial
structure of the data to build clusters and perform the inference at the cluster level.
We first show how to use clustered inference with DL (CluDL). We then show how to make
clustered inference with Ensembled CluDL (EnCluDL).
"""

# %%
# Generating the data
# -------------------
# We start by simulating 2D data in which the support, i.e., the predictive features,
# is composed of four regions of neighboring pixels located in each corner of the
# 2D image. The target :math:`y` is a continuous variable generated from a linear model.
# To make the problem more challenging, the pixels are spatially correlated using a
# Gaussian filter.

import matplotlib.pyplot as plt
import seaborn as sns
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
print(f"Number of samples: {X_init.shape[0]}, Number of features: {n_features}")


# visualize the data
cmap = ListedColormap(["white", "tab:green"])  # 0 -> blue (null), 1 -> green (support)
fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"xticks": [], "yticks": []})
ax.imshow(
    beta.reshape(shape),
    cmap=cmap,
    vmin=0,
    vmax=1,
)

# Legend: green = support, blue = null
legend_handles = [
    Patch(facecolor="tab:green", edgecolor="k", label="Support"),
    Patch(facecolor="white", edgecolor="k", label="Null"),
]
ax.legend(handles=legend_handles, loc="lower center")
plt.tight_layout()


# %%
# Inference with Desparsified Lasso
# ---------------------------------
# First, we perform inference using the Desparsified Lasso method, we treat the data
# as a standard high-dimensional regression problem without taking into account
# the spatial structure of the data. At the inference step, we aim at recovering the
# support while controlling the Family Wise Error Rate (FWER) at a targeted level of
# 0.1. To do so, we use Bonferroni correction by a factor equal to the number of
# features. For more details about the Desparsified Lasso method, see :
# footcite:t:`javanmard2014confidence`, :footcite:p:`zhang2014confidence`
# and :footcite:p:`van2014asymptotically`

import numpy as np
from sklearn.linear_model import LassoCV

from hidimstat import DesparsifiedLasso

fwer_target = 0.1
n_jobs = 5  # number of workers for parallel computing

# compute desparsified lasso
estimator = LassoCV(max_iter=1000, tol=0.0001, eps=0.01, fit_intercept=False)
dl = DesparsifiedLasso(estimator=estimator, n_jobs=n_jobs, random_state=0)
dl.fit_importance(X_init, y)

# compute estimated support (first method)
selected_dl = (dl.pvalues_ < (fwer_target / 2) / n_features) | (
    dl.one_minus_pvalues_ < (fwer_target / 2) / n_features
)

tp_mask = ((selected_dl.astype(int) == 1) & (beta == 1)).astype(bool)
fp_mask = ((selected_dl.astype(int) == 1) & (beta == 0)).astype(bool)
fn_mask = ((selected_dl.astype(int) == 0) & (beta == 1)).astype(bool)
mask_dl = np.zeros(shape)
mask_dl[tp_mask.reshape(shape)] = 1
mask_dl[fp_mask.reshape(shape)] = -2
mask_dl[fn_mask.reshape(shape)] = -1


# %%
# We visualize the support estimated by the Desparsified Lasso method.

_, ax = plt.subplots(figsize=(4, 4), subplot_kw={"xticks": [], "yticks": []})
cmap = ListedColormap(["tab:red", "tab:purple", "white", "tab:green"])
ax.imshow(mask_dl, cmap=cmap, vmin=-2, vmax=1)
legend_handles = [
    Patch(facecolor="tab:green", edgecolor="k", label="True Positive"),
    Patch(facecolor="tab:red", edgecolor="k", label="False Positive"),
    Patch(facecolor="tab:purple", edgecolor="k", label="False Negative"),
]
ax.legend(handles=legend_handles, loc="lower center")
plt.tight_layout()


# %%
# It can be seen that:
# - the Desparsified Lasso method is not powerful enough to recover the support, it
#   only selects scattered pixels as true positives without identifying the regions.
# - the number of false positives is quite high, and sometimes false positives are
#   located far from the true support.


# %%
# Clustered inference with CluDL
# -------------------------------
# To improve the power of the inference, we can leverage the spatial structure
# of the data. The idea is to group correlated pixels into clusters with an additional
# spatial constraint (pixels are iteratively merged with neighboring pixels). This
# leads to a dimension reduction while preserving the spatial structure of the data. To
# control the FWER at the targeted level of 0.1, we perform Bonferroni correction
# but here the correction factor is equal to the number of clusters instead of the
# number of features. For more details about CluDL, see
# :footcite:t:`chevalier2022spatially`

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

selected_cdl = (clu_dl.pvalues_ < (fwer_target / 2) / n_clusters) | (
    clu_dl.one_minus_pvalues_ < (fwer_target / 2) / n_clusters
)


# %%
# Visualizing the support estimated by CluDL

tp_mask = ((selected_cdl.astype(int) == 1) & (beta == 1)).astype(bool)
fp_mask = ((selected_cdl.astype(int) == 1) & (beta == 0)).astype(bool)
fn_mask = ((selected_cdl.astype(int) == 0) & (beta == 1)).astype(bool)
mask_cludl = np.zeros(shape)
mask_cludl[tp_mask.reshape(shape)] = 1
mask_cludl[fp_mask.reshape(shape)] = -2
mask_cludl[fn_mask.reshape(shape)] = -1

_, ax = plt.subplots(figsize=(4, 4), subplot_kw={"xticks": [], "yticks": []})
cmap = ListedColormap(["tab:red", "tab:purple", "white", "tab:green"])
ax.imshow(mask_cludl, cmap=cmap, vmin=-2, vmax=1)
legend_handles = [
    Patch(facecolor="tab:green", edgecolor="k", label="True Positive"),
    Patch(facecolor="tab:red", edgecolor="k", label="False Positive"),
    Patch(facecolor="tab:purple", edgecolor="k", label="False Negative"),
]
ax.legend(handles=legend_handles, loc="lower center")
plt.tight_layout()


# %%
# Inference with Ensembled CluDL
# ------------------------------
# Finally, we perform inference using an ensembled version of CluDL called
# EnCluDL. The idea is to run several CluDL algorithms with different clustering
# choices. By repeating the clustering step, on different bootstrap samples of the data,
# this approach derandomizes the clustering choice and makes it more robust to small
# variations in the data. It can be efficiently parallelized since the different CluDL
# runs are independent and thus embarrassingly parallel. Similar to CluDL, we perform
# Bonferroni correction with a factor equal to the number of clusters on the pvalues
# obtained by aggregating the different CluDL runs. For more details about EnCluDL,
# see :footcite:t:`chevalier2022spatially`

from hidimstat.ensemble_clustered_inference import EnCluDL

enclu_dl = EnCluDL(
    desparsified_lasso=DesparsifiedLasso(estimator=clone(estimator), n_jobs=1),
    clustering=clustering,
    n_bootstraps=20,
    random_state=0,
    n_jobs=n_jobs,
    cluster_boostrap_size=0.5,
)
enclu_dl.fit_importance(X_init, y)

selected_ecdl = (enclu_dl.pvalues_ < (fwer_target / 2) / n_clusters) | (
    enclu_dl.one_minus_pvalues_ < (fwer_target / 2) / n_clusters
)


# %%
# Visualizing the support estimated by EnCluDL

tp_mask = ((selected_ecdl.astype(int) == 1) & (beta == 1)).astype(bool)
fp_mask = ((selected_ecdl.astype(int) == 1) & (beta == 0)).astype(bool)
fn_mask = ((selected_ecdl.astype(int) == 0) & (beta == 1)).astype(bool)
mask_encludl = np.zeros(shape)
mask_encludl[tp_mask.reshape(shape)] = 1
mask_encludl[fp_mask.reshape(shape)] = -2
mask_encludl[fn_mask.reshape(shape)] = -1


_, ax = plt.subplots(figsize=(4, 4), subplot_kw={"xticks": [], "yticks": []})
cmap = ListedColormap(["tab:red", "tab:purple", "white", "tab:green"])
ax.imshow(mask_encludl, cmap=cmap, vmin=-2, vmax=1)
legend_handles = [
    Patch(facecolor="tab:green", edgecolor="k", label="True Positive"),
    Patch(facecolor="tab:red", edgecolor="k", label="False Positive"),
    Patch(facecolor="tab:purple", edgecolor="k", label="False Negative"),
]
ax.legend(handles=legend_handles, loc="lower center")
plt.tight_layout()


# %%
# Support recovery with spatial tolerance
# ---------------------------------------
# For spatially structured data, it is can also be relevant to consider a spatially
# relaxed support recovery. The idea is to penalize less false discoveries that are
# close to the true support. To do so, we introduce an extended support that
# includes the true support and a tolerance region around it.

spatial_tolerance = 3
roi_size_extended = roi_size + spatial_tolerance
beta_extended = beta.copy().reshape(shape)
beta_extended[0:roi_size_extended, 0:roi_size_extended] += 1
beta_extended[-roi_size_extended:, -roi_size_extended:] += 1
beta_extended[0:roi_size_extended, -roi_size_extended:] += 1
beta_extended[-roi_size_extended:, 0:roi_size_extended] += 1

# visualize the extended support
cmap = ListedColormap(["white", "tab:orange", "tab:green"])
fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"xticks": [], "yticks": []})
ax.imshow(beta_extended.reshape(shape), cmap=cmap, vmin=0, vmax=2)
# Legend: green = support, yellow = tolerance region, white = null
legend_handles = [
    Patch(facecolor="tab:green", edgecolor="k", label="Support"),
    Patch(facecolor="tab:orange", edgecolor="k", label="Tolerance region"),
    Patch(facecolor="white", edgecolor="k", label="Null"),
]
ax.legend(handles=legend_handles, loc="lower center")
plt.tight_layout()


# %%
# Comparison with spatial tolerance
# ---------------------------------
# Now, we compare the three methods presented above (Desparsified Lasso, CluDL and
# EnCluDL) in terms of support recovery with spatial tolerance.


def compute_spatially_relaxed_mask(mask, beta_extended):
    """Mark the false positives that are in the tolerance with a special value, -3"""
    support = beta_extended.reshape(shape) >= 1
    mask[support & (mask == -2)] = -3
    return mask


mask_dl_relaxed = compute_spatially_relaxed_mask(mask_dl, beta_extended)
mask_cludl_relaxed = compute_spatially_relaxed_mask(mask_cludl, beta_extended)
mask_encldl_relaxed = compute_spatially_relaxed_mask(mask_encludl, beta_extended)
cmap = ListedColormap(["tab:orange", "tab:red", "tab:purple", "white", "tab:green"])

# sphinx_gallery_thumbnail_number = 6
_, axes = plt.subplots(1, 3, figsize=(10, 4), subplot_kw={"xticks": [], "yticks": []})
axes[0].imshow(mask_dl_relaxed, cmap=cmap, vmin=-3, vmax=1)
axes[0].set_title("Desparsified Lasso", fontweight="bold")
axes[1].imshow(mask_cludl_relaxed, cmap=cmap, vmin=-3, vmax=1)
axes[1].set_title("CluDL", fontweight="bold")
axes[2].imshow(mask_encldl_relaxed, cmap=cmap, vmin=-3, vmax=1)
axes[2].set_title("EnCluDL", fontweight="bold")
legend_handles = [
    Patch(facecolor="tab:green", edgecolor="k", label="True Positive"),
    Patch(facecolor="tab:orange", edgecolor="k", label="False Positive in tolerance"),
    Patch(facecolor="tab:red", edgecolor="k", label="False Positive out of tolerance"),
    Patch(facecolor="tab:purple", edgecolor="k", label="False Negative"),
]
axes[0].legend(handles=legend_handles, loc="lower center")
plt.tight_layout()


# %%
# Note: Choosing inference parameters
# -----------------------------------
# The choice of the number of clusters depends on several parameters, such as:
# the structure of the data (a higher correlation between neighboring features
# enable a greater dimension reduction, i.e. a smaller number of clusters),
# the number of samples (small datasets require more dimension reduction) and
# the required spatial tolerance (small clusters lead to limited spatial
# uncertainty). Formally, "spatial tolerance" is defined by the largest
# distance from the true support for which the occurrence of a false discovery
# is not statistically controlled (c.f. :footcite:t:`chevalier2022spatially`).
# Theoretically, the spatial tolerance ``delta`` is equal to the largest
# cluster diameter. However this choice is conservative, notably in the case
# of ensembled clustered inference. For these algorithms, we recommend to take
# the average cluster radius. In this example, we choose ``n_clusters = 200``,
# leading to a theoretical spatial tolerance ``delta = 6``, which is still
# conservative (see Results).


# %%
# References
# ----------
# .. footbibliography::
