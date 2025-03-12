"""
Support recovery on MEG data
============================

This example compares several methods that recover the support in the MEG/EEG
source localization problem with statistical guarantees. Here we work
with two datasets that study three different tasks (visual, audio, somato).

We reproduce the real data experiment of Chevalier et al. (2020) [1]_,
which shows the benefit of (ensemble) clustered inference such as
(ensemble of) clustered desparsified Multi-Task Lasso ((e)cd-MTLasso)
over standard approach such as sLORETA. Specifically, it retrieves
the support using a natural threshold (not computed a posteriori)
of the estimated parameter. The estimated support enjoys statistical
guarantees.

References
----------
.. [1] Chevalier, J. A., Gramfort, A., Salmon, J., & Thirion, B. (2020).
       Statistical control for spatio-temporal MEG/EEG source imaging with
       desparsified multi-task Lasso. In NeurIPS 2020-34h Conference on
       Neural Information Processing Systems.
"""

import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.datasets import sample, somato
from mne.inverse_sparse.mxne_inverse import _make_sparse_stc, _prepare_gain
from mne.minimum_norm import apply_inverse, make_inverse_operator
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics.pairwise import pairwise_distances

from hidimstat.clustered_inference import clustered_inference
from hidimstat.ensemble_clustered_inference import ensemble_clustered_inference
from hidimstat.statistical_tools.pvalues import zscore_from_pval

##############################################################################
# Specific preprocessing functions
# --------------------------------
# The functions below are used to load or preprocess the data or to put
# the solution in a convenient format. If you are reading this example
# for the first time, you should skip this section.
#
# The following function loads the data from the sample dataset.


def _load_sample(cond):
    """Load data from the sample dataset"""

    # Get data paths
    subject = "sample"
    data_path = sample.data_path()
    fwd_fname_suffix = "MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif"
    fwd_fname = os.path.join(data_path, fwd_fname_suffix)
    ave_fname = os.path.join(data_path, "MEG/sample/sample_audvis-ave.fif")
    cov_fname_suffix = "MEG/sample/sample_audvis-shrunk-cov.fif"
    cov_fname = os.path.join(data_path, cov_fname_suffix)
    cov_fname = data_path + "/" + cov_fname_suffix
    subjects_dir = os.path.join(data_path, "subjects")

    if cond == "audio":
        condition = "Left Auditory"
    elif cond == "visual":
        condition = "Left visual"

    # Read noise covariance matrix
    noise_cov = mne.read_cov(cov_fname)

    # Read forward matrix
    forward = mne.read_forward_solution(fwd_fname)

    # Handling average file
    evoked = mne.read_evokeds(ave_fname, condition=condition, baseline=(None, 0))
    evoked = evoked.pick_types("grad")

    # Selecting relevant time window
    evoked.plot()
    t_min, t_max = 0.05, 0.1
    t_step = 0.01

    pca = False

    return (
        subject,
        subjects_dir,
        noise_cov,
        forward,
        evoked,
        t_min,
        t_max,
        t_step,
        pca,
    )


##############################################################################
# The next function loads the data from the somato dataset.


def _load_somato(cond):
    """Load data from the somato dataset"""

    # Get data paths
    data_path = somato.data_path()
    subject = "01"
    subjects_dir = data_path / "/derivatives/freesurfer/subjects"
    raw_fname = os.path.join(
        data_path, f"sub-{subject}", "meg", f"sub-{subject}_task-{cond}_meg.fif"
    )
    fwd_fname = os.path.join(
        data_path, "derivatives", f"sub-{subject}", f"sub-{subject}_task-{cond}-fwd.fif"
    )

    # Read evoked
    raw = mne.io.read_raw_fif(raw_fname)
    events = mne.find_events(raw, stim_channel="STI 014")
    reject = dict(grad=4000e-13, eog=350e-6)
    picks = mne.pick_types(raw.info, meg=True, eeg=True, eog=True)

    event_id, tmin, tmax = 1, -0.2, 0.25
    epochs = mne.Epochs(
        raw, events, event_id, tmin, tmax, picks=picks, reject=reject, preload=True
    )
    evoked = epochs.average()
    evoked = evoked.pick_types("grad")
    # evoked.plot()

    # Compute noise covariance matrix
    noise_cov = mne.compute_covariance(epochs, rank="info", tmax=0.0)

    # Read forward matrix
    forward = mne.read_forward_solution(fwd_fname)

    # Selecting relevant time window: focusing on early signal
    t_min, t_max = 0.03, 0.05
    t_step = 1.0 / 300

    # We must reduce the whitener since data were preprocessed for removal
    # of environmental noise with maxwell filter leading to an effective
    # number of 64 samples.
    pca = True

    return (
        subject,
        subjects_dir,
        noise_cov,
        forward,
        evoked,
        t_min,
        t_max,
        t_step,
        pca,
    )


##############################################################################
# The function below preprocess the raw M/EEG data, it notably computes the
# whitened MEG/EEG measurements and prepares the gain matrix.


def preprocess_meg_eeg_data(
    evoked, forward, noise_cov, loose=0.0, depth=0.0, pca=False
):
    """Preprocess MEG or EEG data to produce the whitened MEG/EEG measurements
    (target) and the preprocessed gain matrix (design matrix). This function
    is mainly wrapping the `_prepare_gain` MNE function.

    Parameters
    ----------
    evoked : instance of mne.Evoked
        The evoked data.

    forward : instance of Forward
        The forward solution.

    noise_cov : instance of Covariance
        The noise covariance.

    loose : float in [0, 1] or 'auto'
        Value that weights the source variances of the dipole components
        that are parallel (tangential) to the cortical surface. If loose
        is 0 then the solution is computed with fixed orientation.
        If loose is 1, it corresponds to free orientations.
        The default value ('auto') is set to 0.2 for surface-oriented source
        space and set to 1.0 for volumic or discrete source space.
        See for details:
        https://mne.tools/stable/auto_tutorials/inverse/35_dipole_orientations.html?highlight=loose

    depth : None or float in [0, 1]
        Depth weighting coefficients. If None, no depth weighting is performed.

    pca : bool, optional (default=False)
        If True, whitener is reduced.
        If False, whitener is not reduced (square matrix).

    Returns
    -------
    G : array, shape (n_channels, n_dipoles)
        The preprocessed gain matrix. If pca=True then n_channels is
        effectively equal to the rank of the data.

    M : array, shape (n_channels, n_times)
        The whitened MEG/EEG measurements. If pca=True then n_channels is
        effectively equal to the rank of the data.

    forward : instance of Forward
        The preprocessed forward solution.
    """

    all_ch_names = evoked.ch_names

    # Handle depth weighting and whitening (here is no weights)
    forward, G, gain_info, whitener, _, _ = _prepare_gain(
        forward,
        evoked.info,
        noise_cov,
        pca=pca,
        depth=depth,
        loose=loose,
        weights=None,
        weights_min=None,
        rank=None,
    )

    # Select channels of interest
    sel = [all_ch_names.index(name) for name in gain_info["ch_names"]]

    M = evoked.data[sel]
    M = np.dot(whitener, M)

    return G, M, forward


##############################################################################
# The next function translates the solution in a readable format for the
# MNE plotting functions that require a Source Time Course (STC) object.


def _compute_stc(zscore_active_set, active_set, evoked, forward):
    """Wrapper of `_make_sparse_stc`"""

    X = np.atleast_2d(zscore_active_set)

    if X.shape[1] > 1 and X.shape[0] == 1:
        X = X.T

    stc = _make_sparse_stc(
        X, active_set, forward, tmin=evoked.times[0], tstep=1.0 / evoked.info["sfreq"]
    )
    return stc


##############################################################################
# The function below will be used to modify the connectivity matrix
# to avoid multiple warnings when we run the clustering algorithm.


def _fix_connectivity(X, connectivity, affinity):
    """Complete the connectivity matrix if necessary"""

    # Convert connectivity matrix into LIL format
    connectivity = connectivity.tolil()

    # Compute the number of nodes
    n_connected_components, labels = connected_components(connectivity)

    if n_connected_components > 1:

        for i in range(n_connected_components):
            idx_i = np.where(labels == i)[0]
            Xi = X[idx_i]
            for j in range(i):
                idx_j = np.where(labels == j)[0]
                Xj = X[idx_j]
                D = pairwise_distances(Xi, Xj, metric=affinity)
                ii, jj = np.where(D == np.min(D))
                ii = ii[0]
                jj = jj[0]
                connectivity[idx_i[ii], idx_j[jj]] = True
                connectivity[idx_j[jj], idx_i[ii]] = True

    return connectivity, n_connected_components


##############################################################################
# Downloading data
# ----------------
#
# After choosing a task, we run the function that loads the data to get
# the corresponding evoked, forward and noise covariance matrices.

# Choose the experiment (task)
list_cond = ["audio", "visual", "somato"]
cond = list_cond[2]
print(f"Let's process the condition: {cond}")

# Load the data
if cond in ["audio", "visual"]:
    sub, subs_dir, noise_cov, forward, evoked, t_min, t_max, t_step, pca = _load_sample(
        cond
    )

elif cond == "somato":
    sub, subs_dir, noise_cov, forward, evoked, t_min, t_max, t_step, pca = _load_somato(
        cond
    )

##############################################################################
# Preparing data for clustered inference
# --------------------------------------
#
# For clustered inference we need the targets ``Y``, the design matrix ``X``
# and the ``connectivity`` matrix, which is a sparse adjacency matrix.

# Collecting features' connectivity
connectivity = mne.source_estimate.spatial_src_adjacency(forward["src"])

# Croping evoked according to relevant time window
evoked.crop(tmin=t_min, tmax=t_max)

# Choosing frequency and number of clusters used for compression.
# Reducing the frequency to 100Hz to make inference faster
step = int(t_step * evoked.info["sfreq"])
evoked.decimate(step)
t_min = evoked.times[0]
t_step = 1.0 / evoked.info["sfreq"]

# Preprocessing MEG data
X, Y, forward = preprocess_meg_eeg_data(evoked, forward, noise_cov, pca=pca)

##############################################################################
# Running clustered inference
# ---------------------------
#
# For MEG data ``n_clusters = 1000`` is generally a good default choice.
# Taking ``n_clusters > 2000`` might lead to an unpowerful inference.
# Taking ``n_clusters < 500`` might compress too much the data leading
# to a compressed problem not close enough to the original problem.

n_clusters = 1000

# Setting theoretical FWER target
fwer_target = 0.1

# Computing threshold taking into account for Bonferroni correction
correction_clust_inf = 1.0 / n_clusters
zscore_threshold = zscore_from_pval((fwer_target / 2) * correction_clust_inf)

# Initializing FeatureAgglomeration object used for the clustering step
connectivity_fixed, _ = _fix_connectivity(X.T, connectivity, affinity="euclidean")
ward = FeatureAgglomeration(n_clusters=n_clusters, connectivity=connectivity)

# Making the inference with the clustered inference algorithm
inference_method = "desparsified-group-lasso"
beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = clustered_inference(
    X, Y, ward, n_clusters, method=inference_method
)

# Extracting active set (support)
active_set = np.logical_or(
    pval_corr < fwer_target / 2, one_minus_pval_corr < fwer_target / 2
)
active_set_full = np.copy(active_set)
active_set_full[:] = True

# Translating p-vals into z-scores for nicer visualization
zscore = zscore_from_pval(pval, one_minus_pval)
zscore_active_set = zscore[active_set]

##############################################################################
# Visualization
# -------------
# Now, let us plot the thresholded statistical maps derived thanks to the
# clustered inference algorithm referred as cd-MTLasso.

# Let's put the solution into the format supported by the plotting functions
stc = _compute_stc(zscore_active_set, active_set, evoked, forward)

# Plotting parameters
if cond == "audio":
    hemi = "lh"
    view = "lateral"
elif cond == "visual":
    hemi = "rh"
    view = "medial"
elif cond == "somato":
    hemi = "rh"
    view = "lateral"

# Plotting clustered inference solution
mne.viz.set_3d_backend("pyvista")

if active_set.sum() != 0:
    max_stc = np.max(np.abs(stc.data))
    clim = dict(pos_lims=(3, zscore_threshold, max_stc), kind="value")
    brain = stc.plot(
        subject=sub,
        hemi=hemi,
        clim=clim,
        subjects_dir=subs_dir,
        views=view,
        time_viewer=False,
    )
    brain.add_text(0.05, 0.9, f"{cond} - cd-MTLasso", "title", font_size=20)

# Hack for nice figures on HiDimStat website
save_fig = False
plot_saved_fig = True
if save_fig:
    brain.save_image(f"figures/meg_{cond}_cd-MTLasso.png")
if plot_saved_fig:
    brain.close()
    img = mpimg.imread(f"figures/meg_{cond}_cd-MTLasso.png")
    plt.imshow(img)
    plt.axis("off")
    plt.show()

interactive_plot = False
if interactive_plot:
    brain = stc.plot(subject=sub, hemi="both", subjects_dir=subs_dir, clim=clim)

##############################################################################
# Comparision with sLORETA
# ------------------------
# Now, we compare the results derived from cd-MTLasso with the solution
# obtained from the one of the most standard approach: sLORETA.

# Running sLORETA with standard hyper-parameter
lambda2 = 1.0 / 9
inv = make_inverse_operator(
    evoked.info, forward, noise_cov, loose=0.0, depth=0.0, fixed=True
)
stc_full = apply_inverse(evoked, inv, lambda2=lambda2, method="sLORETA")
stc_full = stc_full.mean()

# Computing threshold taking into account for Bonferroni correction
n_features = stc_full.data.size
correction = 1.0 / n_features
zscore_threshold_no_clust = zscore_from_pval((fwer_target / 2) * correction)

# Computing estimated support by sLORETA
active_set = np.abs(stc_full.data) > zscore_threshold_no_clust
active_set = active_set.flatten()

# Putting the solution into the format supported by the plotting functions
sLORETA_solution = np.atleast_2d(stc_full.data[active_set]).flatten()
stc = _make_sparse_stc(
    sLORETA_solution, active_set, forward, stc_full.tmin, tstep=stc_full.tstep
)

# Plotting sLORETA solution
if active_set.sum() != 0:
    max_stc = np.max(np.abs(stc.data))
    clim = dict(pos_lims=(3, zscore_threshold_no_clust, max_stc), kind="value")
    brain = stc.plot(
        subject=sub,
        hemi=hemi,
        clim=clim,
        subjects_dir=subs_dir,
        views=view,
        time_viewer=False,
    )
    brain.add_text(0.05, 0.9, f"{cond} - sLORETA", "title", font_size=20)

    # Hack for nice figures on HiDimStat website
    if save_fig:
        brain.save_image(f"figures/meg_{cond}_sLORETA.png")
    if plot_saved_fig:
        brain.close()
        img = mpimg.imread(f"figures/meg_{cond}_sLORETA.png")
        plt.imshow(img)
        plt.axis("off")
        plt.show()

##############################################################################
# Analysis of the results
# -----------------------
# While the clustered inference solution always highlights the expected
# cortex (audio, visual or somato-sensory) with a universal predertemined
# threshold, the solution derived from the sLORETA method does not enjoy
# the same property. For the audio task the method is conservative and
# for the somato task the method makes false discoveries (then it seems
# anti-conservative).


##############################################################################
# Running ensemble clustered inference
# ------------------------------------
#
# To go further it is possible to run the ensemble clustered inference
# algorithm. It might take several minutes on standard device with
# ``n_jobs=1`` (around 10 min). Just set
# ``run_ensemble_clustered_inference=True`` below.
run_ensemble_clustered_inference = False

if run_ensemble_clustered_inference:
    # Making the inference with the ensembled clustered inference algorithm
    beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = (
        ensemble_clustered_inference(
            X, Y, ward, n_clusters, inference_method=inference_method
        )
    )

    # Extracting active set (support)
    active_set = np.logical_or(
        pval_corr < fwer_target / 2, one_minus_pval_corr < fwer_target / 2
    )
    active_set_full = np.copy(active_set)
    active_set_full[:] = True

    # Translating p-vals into z-scores for nicer visualization
    zscore = zscore_from_pval(pval, one_minus_pval)
    zscore_active_set = zscore[active_set]

    # Putting the solution into the format supported by the plotting functions
    stc = _compute_stc(zscore_active_set, active_set, evoked, forward)

    # Plotting ensemble clustered inference solution
    if active_set.sum() != 0:
        max_stc = np.max(np.abs(stc._data))
        clim = dict(pos_lims=(3, zscore_threshold, max_stc), kind="value")
        brain = stc.plot(
            subject=sub,
            hemi=hemi,
            clim=clim,
            subjects_dir=subs_dir,
            views=view,
            time_viewer=False,
        )
        brain.add_text(0.05, 0.9, f"{cond} - ecd-MTLasso", "title", font_size=20)

        # Hack for nice figures on HiDimStat website
        if save_fig:
            brain.save_image(f"figures/meg_{cond}_ecd-MTLasso.png")
        if plot_saved_fig:
            brain.close()
            img = mpimg.imread(f"figures/meg_{cond}_ecd-MTLasso.png")
            plt.imshow(img)
            plt.axis("off")
            plt.show()

        if interactive_plot:
            brain = stc.plot(subject=sub, hemi="both", subjects_dir=subs_dir, clim=clim)
