"""
Support recovery on fMRI data
=============================

This example compares methods based on Desparsified Lasso (DL) that estimate
a decoder map support with statistical guarantees. Here, we work with the Haxby
dataset and we focus on the 'face vs house' contrast. Thus, we consider the labelled
activation maps of a given subject and try to produce a brain map that
corresponds to the discriminative pattern that makes the decoding of the
two conditions.

In this example, we show that in too high dimension (too many voxels),
DL is not suitable due to memory issues. However, it's possible to overcome
this limitation using aggregation methods based on the structure of the data
(too much correlation between neighbouring voxels).
We present two methods for aggregation features that offer statistical guarantees
but with a (small) spatial tolerance on the shape of the support:
clustered desparsified lasso (CLuDL) combines clustering (parcellation)
and statistical inference ; ensemble of clustered desparsified lasso (EnCluDL)
adds a randomization step over the choice of clustering.

EnCluDL is powerful and does not depend on a unique clustering choice.
As shown in :footcite:t:`chevalier2021decoding`, for several tasks, the estimated
support (predictive regions) looks relevant.
"""

#############################################################################
# Imports needed for this script
# ------------------------------
import resource
import warnings

import numpy as np
import pandas as pd
from matplotlib.pyplot import get_cmap
from nilearn import datasets
from nilearn.image import mean_img
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_stat_map, show
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import image
from sklearn.utils import Bunch

from hidimstat.ensemble_clustered_inference import (
    clustered_inference,
    clustered_inference_pvalue,
)
from hidimstat.ensemble_clustered_inference import (
    ensemble_clustered_inference,
    ensemble_clustered_inference_pvalue,
)
from hidimstat.desparsified_lasso import (
    desparsified_lasso,
    desparsified_lasso_pvalue,
)
from hidimstat.statistical_tools.p_values import zscore_from_pval


# Remmove warnings during loading data
warnings.filterwarnings(
    "ignore", message="The provided image has no sform in its header."
)

# Limit the ressoruce use for the example to 5 G.
resource.setrlimit(resource.RLIMIT_AS, (int(5 * 1e9), int(5 * 1e9)))
n_job = 1


#############################################################################
# Function to fetch and preprocess Haxby dataset
# ----------------------------------------------
def preprocess_haxby(subject=2, memory=None):
    """Gathering and preprocessing Haxby dataset for a given subject."""

    # Gathering data
    haxby_dataset = datasets.fetch_haxby(subjects=[subject])
    fmri_filename = haxby_dataset.func[0]

    behavioral = pd.read_csv(haxby_dataset.session_target[0], sep=" ")

    conditions = behavioral["labels"].values
    session_label = behavioral["chunks"].values

    condition_mask = np.logical_or(conditions == "face", conditions == "house")
    groups = session_label[condition_mask]

    # Loading anatomical image (back-ground image)
    if haxby_dataset.anat[0] is None:
        bg_img = None
    else:
        bg_img = mean_img(haxby_dataset.anat, copy_header=True)

    # Building target where '1' corresponds to 'face' and '-1' to 'house'
    y = np.asarray((conditions[condition_mask] == "face") * 2 - 1)

    # Loading mask
    mask_img = haxby_dataset.mask
    masker = NiftiMasker(
        mask_img=mask_img,
        standardize="zscore_sample",
        smoothing_fwhm=None,
        memory=memory,
    )

    # Computing masked data
    fmri_masked = masker.fit_transform(fmri_filename)
    X = np.asarray(fmri_masked)[condition_mask, :]

    return Bunch(X=X, y=y, groups=groups, bg_img=bg_img, masker=masker)


#############################################################################
# Gathering and preprocessing Haxby dataset for a given subject
# -------------------------------------------------------------
# The `preprocess_haxby` function make the preprocessing of the Haxby dataset,
# it outputs the preprocessed activation maps for the two conditions
# 'face' or 'house' (contained in `X`), the conditions (in `y`),
# the session labels (in `groups`) and the mask (in `masker`).
# You may choose a subject in [1, 2, 3, 4, 5, 6]. By default subject=2.
data = preprocess_haxby(subject=2)
X, y, groups, masker = data.X, data.y, data.groups, data.masker
mask = masker.mask_img_.get_fdata().astype(bool)

#############################################################################
# Initializing FeatureAgglomeration object that performs the clustering
# -------------------------------------------------------------------------
# For fMRI data taking 500 clusters is generally a good default choice.

n_clusters = 500
# Deriving voxels connectivity.
shape = mask.shape
n_x, n_y, n_z = shape[0], shape[1], shape[2]
connectivity = image.grid_to_graph(n_x=n_x, n_y=n_y, n_z=n_z, mask=mask)
# Initializing FeatureAgglomeration object.
ward = FeatureAgglomeration(n_clusters=n_clusters, connectivity=connectivity)

#############################################################################
# Making the inference with several algorithms
# --------------------------------------------

#############################################################################
# First, we try to recover the discriminative pattern by computing
# p-values from desparsified lasso.
# Due to the size of the X, it's not possible to use this method with a limit
# of 5 G for memory. To handle this problem, the following methods use some
# feature aggregation methods.
try:
    beta_hat, sigma_hat, precision_diagonal = desparsified_lasso(
        X, y, noise_method="median", max_iteration=1000
    )
    pval_dl, _, one_minus_pval_dl, _, cb_min, cb_max = desparsified_lasso_pvalue(
        X.shape[0], beta_hat, sigma_hat, precision_diagonal
    )
except MemoryError as err:
    pval_dl = None
    one_minus_pval_dl = None
    print("As expected, Desparsified Lasso uses too much memory.")

#############################################################################
# Now, the clustered inference algorithm which combines parcellation
# and high-dimensional inference (c.f. References).
ward_, beta_hat, theta_hat, omega_diag = clustered_inference(
    X, y, ward, n_clusters, scaler_sampling=StandardScaler(), tolerance=1e-2
)
beta_hat, pval_cdl, _, one_minus_pval_cdl, _ = clustered_inference_pvalue(
    X.shape[0], None, ward_, beta_hat, theta_hat, omega_diag
)

#############################################################################
# Below, we run the ensemble clustered inference algorithm which adds a
# randomization step over the clustered inference algorithm (c.f. References).
# To make the example as short as possible we take `n_bootstraps=5`
# which means that 5 different parcellations are considered and
# then 5 statistical maps are produced and aggregated into one.
# However you might benefit from clustering randomization taking
# `n_bootstraps=25` or `n_bootstraps=100`, also we set `n_jobs=2`.
list_ward, list_beta_hat, list_theta_hat, list_omega_diag = (
    ensemble_clustered_inference(
        X,
        y,
        ward,
        n_clusters,
        groups=groups,
        scaler_sampling=StandardScaler(),
        n_bootstraps=5,
        max_iteration=6000,
        tolerance=1e-2,
        n_jobs=2,
    )
)
beta_hat, selected = ensemble_clustered_inference_pvalue(
    X.shape[0],
    False,
    list_ward,
    list_beta_hat,
    list_theta_hat,
    list_omega_diag,
    fdr=0.1,
)

#############################################################################
# Plotting the results
# --------------------
# To allow a better visualization of the disciminative pattern we will plot
# z-maps rather than p-value maps. Assuming Gaussian distribution of the
# estimators we can recover a z-score from a p-value by using the
# inverse survival function.
#
# First, we set theoretical FWER target at 10%.

n_samples, n_features = X.shape
target_fwer = 0.1

#############################################################################
# We now translate the FWER target into a z-score target.
# For the permutation test methods we do not need any additional correction
# since the p-values are already adjusted for multiple testing.

zscore_threshold_corr = zscore_from_pval((target_fwer / 2))

#############################################################################
# Other methods need to be corrected. We consider the Bonferroni correction.
# For methods that do not reduce the feature space, the correction
# consists in dividing by the number of features.

correction = 1.0 / n_features
zscore_threshold_no_clust = zscore_from_pval((target_fwer / 2) * correction)

#############################################################################
# For methods that parcelates the brain into groups of voxels, the correction
# consists in dividing by the number of parcels (or clusters).

correction_clust = 1.0 / n_clusters
zscore_threshold_clust = zscore_from_pval((target_fwer / 2) * correction_clust)

#############################################################################
# Now, we can plot the thresholded z-score maps by translating the
# p-value maps estimated previously into z-score maps and using the
# suitable threshold. For a better readability, we make a small function
# called `plot_map` that wraps all these steps.


def plot_map(
    data,
    threshold,
    title=None,
    cut_coords=[-25, -40, -5],
    masker=masker,
    bg_img=data.bg_img,
    vmin=None,
    vmax=None,
):
    zscore_img = masker.inverse_transform(data)
    plot_stat_map(
        zscore_img,
        threshold=threshold,
        bg_img=bg_img,
        dim=-1,
        cut_coords=cut_coords,
        title=title,
        cmap=get_cmap("bwr"),
        vmin=vmin,
        vmax=vmax,
    )


if pval_dl is not None:
    plot_map(
        zscore_from_pval(pval_dl, one_minus_pval_dl),
        zscore_threshold_no_clust,
        "Desparsified Lasso",
    )

plot_map(
    zscore_from_pval(pval_cdl, one_minus_pval_cdl), zscore_threshold_clust, "CluDL"
)

plot_map(selected, 0.5, "EnCluDL", vmin=-1, vmax=1)

#############################################################################
# Analysis of the results
# -----------------------
# As advocated in introduction, the methods that do not reduce the original
# problem are not satisfying since they are too conservative.
# Among those methods, the only one that makes discoveries is the one that
# threshold the SVR decoder using a parametric approximation.
# However this method has no statistical guarantees and we can see that some
# isolated voxels are discovered, which seems quite spurious.
# The discriminative pattern derived from the clustered inference algorithm
# (CluDL) show that the method is less conservative.
# However, some reasonable paterns are also included in this solution.
# Finally, the solution provided by the ensemble clustered inference algorithm
# (EnCluDL) seems realistic as we recover the visual cortex and do not make
# spurious discoveries.

show()

#############################################################################
# References
# ----------
# .. footbibliography::
