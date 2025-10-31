r"""
Pixel-wise inference on digit classification
============================================

This example illustrates how to perform pixel-wise inference using Ensemble Clustered
Inference with Desparsified Lasso (EnCluDL) on digit classification tasks. We use the
MNIST dataset and consider binary classification between digits 1 vs 7 and 0 vs 1. The
MNIST dataset contains 28x28 pixel images of handwritten digits.
"""

# %%
# Loading the MNIST dataset
# -------------------------
# We start by loading the MNIST dataset from OpenML. We then filter the dataset to
# include only digits 1 and 7 for the first classification task, and digits 0 and 1
# for the second task. Finally, we visualize a few samples from each class.


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

mnist_dataset = fetch_openml("mnist_784", version=1, as_frame=False)
X_mnist, y_mnist = mnist_dataset.data, mnist_dataset.target
mask_1_7 = (y_mnist == "1") | (y_mnist == "7")
X_1_7, y_1_7 = X_mnist[mask_1_7], y_mnist[mask_1_7].astype(int)

mask_0_1 = (y_mnist == "0") | (y_mnist == "1")
X_0_1, y_0_1 = X_mnist[mask_0_1], y_mnist[mask_0_1].astype(int)


_, axes = plt.subplots(2, 5, figsize=(8, 4))
# First row - digits 0 and 1
for i in range(5):
    label = 1 if i % 2 == 0 else 0
    axes[0, i].imshow(X_0_1[y_0_1 == label][i].reshape(28, 28), cmap="gray")
    axes[0, i].axis("off")
axes[0, 2].set_title("Digits 0 vs 1", fontweight="bold", y=1.05)

# Second row - digits 1 and 7
for i in range(5):
    label = 7 if i % 2 == 0 else 1
    axes[1, i].imshow(X_1_7[y_1_7 == label][i].reshape(28, 28), cmap="gray")
    axes[1, i].axis("off")
axes[1, 2].set_title("Digits 1 vs 7", fontweight="bold", y=1.05)
_ = plt.tight_layout()


# %%
# Clustering pixels using connectivity graph
# ------------------------------------------
# We create a connectivity graph based on the 2D grid structure of the images using
# the `grid_to_graph` function from `sklearn.feature_extraction.image`. We then apply
# Ward hierarchical clustering, implemented in `FeatureAgglomeration`, to cluster the
# pixels with connectivity constraints. The resulting clusters contain spatially
# contiguous pixels that are highly correlated (high within-cluster correlation).
# However, the clusters themselves are less correlated with each other (low between-cluster
# correlation). By running inference on these clusters, we can improve the statistical
# power of the inference procedure. We then visualize the clustering results and observe
# that the clusters are spatially contiguous and capture meaningful structures in the
# digit images.

from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_extraction import image
from sklearn.model_selection import train_test_split

connectivity = image.grid_to_graph(n_x=28, n_y=28)

n_clusters = 5
ward_clustering_1_7 = FeatureAgglomeration(
    n_clusters=n_clusters, connectivity=connectivity
)
X_cluster = train_test_split(X_1_7, train_size=0.5, random_state=0)[0]
ward_clustering_1_7.fit(X_cluster)

ward_clustering_0_1 = FeatureAgglomeration(
    n_clusters=n_clusters, connectivity=connectivity
)
X_cluster = train_test_split(X_0_1, train_size=0.5, random_state=0)[0]
ward_clustering_0_1.fit(X_cluster)

_, axes = plt.subplots(1, 2, figsize=(6, 3))
ax = axes[0]
ax.imshow(ward_clustering_1_7.labels_.reshape(28, 28), cmap="Set2")
ax.axis("off")
_ = ax.set_title("Clustering for digits 1 vs 7")

ax = axes[1]
ax.imshow(ward_clustering_0_1.labels_.reshape(28, 28), cmap="Set2")
ax.axis("off")
_ = ax.set_title("Clustering for digits 0 vs 1")

_ = plt.tight_layout()


# %%
# Randomness in clustering affects inference stability
# ----------------------------------------------------
# A key limitation of clustered inference procedures is that the clustering step
# introduces randomness into the inference process. Data perturbations can produce
# different clustering results, which subsequently lead to varying inference outcomes.
# This effect can be demonstrated by running the clustering algorithm multiple times
# on different subsamples of the data.


_, axes = plt.subplots(1, 4, figsize=(9, 3))

for i, ax in enumerate(axes):
    X_cluster = train_test_split(X_1_7, y_1_7, train_size=0.5, random_state=i)[0]
    ward_clustering = FeatureAgglomeration(
        n_clusters=n_clusters, connectivity=connectivity
    )
    ward_clustering.fit(X_cluster)
    ax.imshow(ward_clustering.labels_.reshape(28, 28), cmap="Set2")
    ax.axis("off")
    ax.set_title(f"Clustering {i}")

_ = plt.tight_layout()


# %%
# Ensemble Clustered Inference with Desparsified Lasso
# ----------------------------------------------------
# To mitigate the randomness introduced by clustering, we ensemble the results from
# multiple clustered inference procedures. This approach derandomizes the inference
# procedure and produces more stable results. We use the `ensemble_clustered_inference`
# function for this purpose. While this approach is more computationally intensive than
# single clustered inference, the procedure can be parallelized across clustering
# repetitions using the `n_jobs` parameter.

from sklearn.preprocessing import StandardScaler

from hidimstat import ensemble_clustered_inference
from hidimstat.ensemble_clustered_inference import ensemble_clustered_inference_pvalue

n_jobs = 5
fdr = 0.5

ward_1_7, desparsified_lassos_1_7 = ensemble_clustered_inference(
    X_1_7,
    y_1_7,
    ward_clustering,
    scaler_sampling=StandardScaler(),
    random_state=0,
    n_jobs=n_jobs,
)
ward_0_1, desparsified_lassos_0_1 = ensemble_clustered_inference(
    X_0_1,
    y_0_1,
    ward_clustering,
    scaler_sampling=StandardScaler(),
    random_state=0,
    n_jobs=n_jobs,
)

beta_hat_1_7, selected_ecdl_1_7 = ensemble_clustered_inference_pvalue(
    X_1_7.shape[0], False, ward_1_7, desparsified_lassos_1_7, fdr=fdr
)
beta_hat_0_1, selected_ecdl_0_1 = ensemble_clustered_inference_pvalue(
    X_0_1.shape[0], False, ward_0_1, desparsified_lassos_0_1, fdr=fdr
)


# %%
# Visualizing the inference results
# ---------------------------------
# We visualize the inference results by plotting the coefficients estimated by the
# desparsified lasso, which have been ensembled over multiple clusterings.

# sphinx_gallery_thumbnail_number = 4

_, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, beta_hat, title in zip(
    axes,
    [beta_hat_0_1, beta_hat_1_7],
    ["Digits 0 vs 1", "Digits 1 vs 7"],
):
    max_abs = np.max(np.abs(beta_hat))
    im = ax.imshow(beta_hat.reshape(28, 28), cmap="RdBu", vmin=-max_abs, vmax=max_abs)
    plt.colorbar(im, label=r"$\hat{\beta}$", ax=ax, shrink=0.7)
    ax.set_title(f"{title}")
    ax.axis("off")

_ = plt.tight_layout()

# %%
# We can also visualize the p-values obtained by aggregating results from multiple
# clustered desparsified lasso procedures. These aggregated p-values provide a more
# robust measure of feature significance compared to single clustering approaches.

_, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, selected_ecdl, title in zip(
    axes,
    [selected_ecdl_0_1, selected_ecdl_1_7],
    ["Digits 0 vs 1", "Digits 1 vs 7"],
):
    im = ax.imshow(selected_ecdl.reshape(28, 28), cmap="RdBu", vmin=0, vmax=1)
    plt.colorbar(im, label="p-value", ax=ax, shrink=0.7)
    ax.set_title(f"Aggregated {title}")
    ax.axis("off")

_ = plt.tight_layout()


# %%
# References
# ----------
# TODO
