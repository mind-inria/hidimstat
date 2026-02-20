"""
Source localization of somatosensory MEG data
=============================================

This example adapts the experiment presented in
:footcite:t:`chevalier2020statistical`. We show how to identify which cortical
sources are activated during a somatosensory task. To do so, we leverage
spatially constrained clustering to effectively reduce the dimensionality of
the problem while accounting for the spatial structure of the data. We then
perform inference with the desparsified multitask Lasso to perform support
recovery from spatio-temporal data.
"""

# %%
# Load somatosensory MEG data from MNE-Python
# -------------------------------------------
# We use the somatosensory MEG dataset available in MNE-Python. We visualize
# the evoked response across MEG sensors.

import mne
import numpy as np
from mne.datasets import somato

cond = "somato"
data_path = somato.data_path(verbose=True)
subject = "01"
subjects_dir = data_path / "derivatives/freesurfer/subjects"
raw_fname = (
    data_path / f"sub-{subject}" / "meg" / f"sub-{subject}_task-{cond}_meg.fif"
)
fwd_fname = (
    data_path
    / "derivatives"
    / f"sub-{subject}"
    / f"sub-{subject}_task-{cond}-fwd.fif"
)

# Read evoked
raw = mne.io.read_raw_fif(raw_fname)
events = mne.find_events(raw, stim_channel="STI 014")
reject = {"grad": 4000e-13, "eog": 350e-6}
picks = mne.pick_types(raw.info, meg=True, eeg=True, eog=True)

event_id, tmin, tmax = 1, -0.2, 0.25
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    picks=picks,
    reject=reject,
    preload=True,
)
evoked = epochs.average()
evoked = evoked.pick_types("grad")
evoked.plot()


# %%
# Preprocessing MEG data for source localization
# ----------------------------------------------
# Before performing source localization, we need to preprocess the MEG data.
# To do so we rely on the MNE-Python library. We compute the forward model,
# which describes how the activity of cortical sources is projected onto MEG
# sensors. We also select the time window of interest, which is the early
# response. Finally we compute the noise covariance matrix, in order to whiten
# the data.

from mne.inverse_sparse.mxne_inverse import _prepare_gain

# Read forward matrix
forward = mne.read_forward_solution(fwd_fname)
# Compute noise covariance matrix
noise_cov = mne.compute_covariance(epochs, rank="info", tmax=0.0)
# We must reduce the whitener since data were preprocessed for removal
# of environmental noise with maxwell filter leading to an effective
# number of 64 samples.
pca = True
# Preprocessing MEG data
forward, G, gain_info, whitener, _, _ = _prepare_gain(
    forward,
    evoked.info,
    noise_cov,
    pca=pca,
    depth=0.0,
    loose=0.0,
    weights=None,
    weights_min=None,
    rank=None,
)

# Selecting relevant time window: focusing on early signal
t_min, t_max = 0.01, 0.05
t_step = 1.0 / 300
# Croping evoked according to relevant time window
evoked.crop(tmin=t_min, tmax=t_max)
# Choosing frequency and number of clusters used for compression.
# Reducing the frequency to 100Hz to make inference faster
step = int(t_step * evoked.info["sfreq"])
evoked.decimate(step)

y = np.dot(whitener, evoked.data)


# %%
# Spatially constrained clustering
# --------------------------------
# We then extract the spatial adjacency matrix that is then used in the
# clustering step to incorporate the spatial structure of the data.
# For MEG data ``n_clusters = 750`` is generally a good default choice.
# Taking ``n_clusters > 2000`` might lead to an unpowerful inference.
# Taking ``n_clusters < 500`` might compress too much the data leading
# to a compressed problem not close enough to the original problem.

from sklearn.cluster import FeatureAgglomeration

# Collecting features' connectivity
connectivity = mne.source_estimate.spatial_src_adjacency(forward["src"])

n_clusters = 750
ward = FeatureAgglomeration(n_clusters=n_clusters, connectivity=connectivity)


# %%
# Running clustered inference
# ---------------------------
# We can now run the clustered inference using the desparsified multitask Lasso
# We then select the clusters that are significant at the target FWER level,
# which is set to 0.1 in this example.

from sklearn.linear_model import MultiTaskLassoCV

from hidimstat import CluDL, DesparsifiedLasso

# Setting theoretical FWER target
fwer_target = 0.1

cludl = CluDL(
    clustering=ward,
    desparsified_lasso=DesparsifiedLasso(estimator=MultiTaskLassoCV()),
    random_state=0,
)

cludl.fit_importance(G, y)
selected = cludl.fwer_selection(fwer_target, two_tailed_test=True)
# multiplying the -log10(p-value) map by the sign of the estimated coefficients
log_pvalues = -np.log10(cludl.pvalues_) * selected


# %%
# Visualize the p-value map on the brain surface
# ----------------------------------------------
# We then visualize the identified cortical sources on the brain surface. We
# plot the -log10(p-value) map to which we assign the sign of the estimated
# coefficient in the multitask Lasso model.

from pathlib import Path

import matplotlib.pyplot as plt
from nilearn import datasets, plotting

stc_pvals = mne.SourceEstimate(
    data=log_pvalues[:, np.newaxis],
    vertices=[forward["src"][0]["vertno"], forward["src"][1]["vertno"]],
    tmin=0.0,
    tstep=1.0,
    subject=f"{subject}",
)

# TODO: tmp fix: unlinking fsaverage directory to avoid error
fs_dir = Path(subjects_dir) / "fsaverage"
if fs_dir.is_symlink():
    fs_dir.unlink()
mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir)

morph = mne.compute_source_morph(
    src=forward["src"],
    subject_from=f"{subject}",
    subject_to="fsaverage",
    subjects_dir=subjects_dir,
    spacing=5,
)
stc_fsaverage = morph.apply(stc_pvals)

start_id = stc_fsaverage.data.shape[0] // 2
rh_stat_map = stc_fsaverage.data[start_id:, 0]

fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")


fig, axes = plt.subplots(
    1,
    2,
    subplot_kw={"projection": "3d"},
    figsize=(8, 5),
    gridspec_kw={"hspace": -0.1},
)

# sphinx_gallery_thumbnail_number = 2
for ax, view in zip(axes, ["lateral", "medial"], strict=True):
    plotting.plot_surf_stat_map(
        surf_mesh=fsaverage.infl_right,
        stat_map=rh_stat_map,
        bg_map=fsaverage.sulc_right,
        hemi="right",
        view=view,
        colorbar=True,
        cmap="RdBu_r",
        threshold=-np.log10(fwer_target),
        vmax=10,
        axes=ax,
        title=f"rh-{view} view",
    )

ax._colorbars[0].set_ylabel("-log10(p-value)")
plotting.show()

# %%
# Most of the identified sources are located in the somatosensory cortex,
# which is expected for this somatosensory task.

# %%
# References
# ----------
# .. footbibliography::
