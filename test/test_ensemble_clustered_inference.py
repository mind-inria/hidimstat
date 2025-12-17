"""
Test the clustered_inference module
"""

import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_extraction import image
from sklearn.linear_model import LassoCV, MultiTaskLassoCV

from hidimstat import CluDL, DesparsifiedLasso, EnCluDL
from hidimstat._utils.scenario import (
    multivariate_simulation,
    multivariate_simulation_spatial,
)
from hidimstat.statistical_tools.multiple_testing import fdp_power


def spatially_relaxed_fdp_power(
    selected, ground_truth, roi_size, spatial_tolerance, shape
):
    """
    Calculate False Discovery Proportion and statistical power with spatial
    relaxation. Useful for testing methods using clustering on spatial data where
    false positives near true positives can be less penalized.

    """
    beta_ids = np.argwhere(ground_truth == 1).flatten()
    roi_size_extended = roi_size + spatial_tolerance
    ground_truth_extended = ground_truth.copy().reshape(shape)
    ground_truth_extended[0:roi_size_extended, 0:roi_size_extended] += 1
    ground_truth_extended[-roi_size_extended:, -roi_size_extended:] += 1
    ground_truth_extended[0:roi_size_extended, -roi_size_extended:] += 1
    ground_truth_extended[-roi_size_extended:, 0:roi_size_extended] += 1
    ground_truth_extended = (ground_truth_extended > 0).astype(int).flatten()

    selected_ids = np.argwhere(selected).flatten()
    true_positive = np.intersect1d(selected_ids, beta_ids)

    ground_truth_extended_ids = np.argwhere(
        ground_truth_extended.flatten() == 1
    ).flatten()
    false_positive = np.setdiff1d(selected_ids, ground_truth_extended_ids)

    fdp = len(false_positive) / len(selected_ids)
    power = len(true_positive) / len(beta_ids)
    return fdp, power


def test_cludl_spatial():
    """
    Test CluDL on a 2D spatial simulation. Testing for support recovery methods using
    clustering is challenging as clusters that intersect the true support can also
    include non-support features, rapidly increasing false positives. To address this,
    we introduce a spatial relaxation in the evaluation metrics.

     - Test that the spatially relaxed FDP is below a specified FDR threshold (0.1).
     - Test that the statistical power is above a specified threshold (0.8).
    """

    n_samples = 200
    shape = (20, 20)
    n_features = shape[1] * shape[0]
    roi_size = 4  # size of the edge of the four predictive regions
    signal_noise_ratio = 32.0  # noise standard deviation
    smooth_X = 0.6  # level of spatial smoothing introduced by the Gaussian filter

    fdp_list = []
    power_list = []
    for seed in range(10):
        # generating the data
        X_init, y, beta, epsilon = multivariate_simulation_spatial(
            n_samples, shape, roi_size, signal_noise_ratio, smooth_X, seed=seed
        )

        y = y - np.mean(y)
        X_init = X_init - np.mean(X_init, axis=0)

        n_clusters = 200
        connectivity = image.grid_to_graph(n_x=n_features, n_y=1, n_z=1)
        clustering = FeatureAgglomeration(
            n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
        )
        estimator = LassoCV(max_iter=1000, tol=0.0001, eps=0.01, fit_intercept=False)
        cludl = CluDL(
            desparsified_lasso=DesparsifiedLasso(estimator=estimator),
            clustering=clustering,
            random_state=seed,
        )
        cludl.fit_importance(X_init, y)
        fdr = 0.1
        selected = cludl.fdr_selection(fdr=fdr, two_tailed_test=False)

        fdp, power = spatially_relaxed_fdp_power(
            selected=selected,
            ground_truth=beta,
            roi_size=roi_size,
            spatial_tolerance=3,
            shape=shape,
        )
        fdp_list.append(fdp)
        power_list.append(power)
    assert np.mean(power_list) >= 0.8
    assert np.mean(fdp_list) <= fdr


def test_cludl_independence():
    """Test that CluDL works with repeated calls
    non-regression test for #425"""
    n_samples = 50
    shape = (20, 20)
    roi_size = 4  # size of the edge of the four predictive regions
    X_init, y, beta, epsilon = multivariate_simulation_spatial(
        n_samples, shape, roi_size, signal_noise_ratio=10., smooth_X=1)
    alpha = .05 # alpha is the significance level for the statistical test
    n_clusters = 50
    connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1])
    ward = FeatureAgglomeration(
        n_clusters=n_clusters, connectivity=connectivity, linkage="ward")

    c1 = CluDL(clustering=ward)
    c1.fit_importance(X_init, y)
    s1 = c1.fwer_selection(alpha, n_tests=n_clusters)
    c2 = CluDL(clustering=ward,cluster_boostrap_size=0.5)
    c2.fit_importance(X_init, y)
    s2 = c2.fwer_selection(alpha, n_tests=n_clusters)
    assert np.sum(s2) > np.sum(s1 / 2)



def test_encludl_spatial():
    """
    Test CluDL on a 2D spatial simulation. Testing for support recovery methods using
    clustering is challenging as clusters that intersect the true support can also
    include non-support features, rapidly increasing false positives. To address this,
    we introduce a spatial relaxation in the evaluation metrics.

     - Test that the spatially relaxed FDP is below a specified FDR threshold (0.1).
     - Test that the statistical power is above a specified threshold (0.8).
    """

    n_samples = 400
    shape = (20, 20)
    n_features = shape[1] * shape[0]
    roi_size = 4  # size of the edge of the four predictive regions
    signal_noise_ratio = 32.0  # noise standard deviation
    smooth_X = 0.6  # level of spatial smoothing introduced by the Gaussian filter

    fdp_list = []
    power_list = []
    for seed in range(10):
        # generating the data
        X_init, y, beta, epsilon = multivariate_simulation_spatial(
            n_samples, shape, roi_size, signal_noise_ratio, smooth_X, seed=seed
        )

        y = y - np.mean(y)
        X_init = X_init - np.mean(X_init, axis=0)

        n_clusters = 200
        connectivity = image.grid_to_graph(n_x=n_features, n_y=1, n_z=1)
        clustering = FeatureAgglomeration(
            n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
        )

        estimator = LassoCV(max_iter=1000, tol=0.0001, eps=0.01, fit_intercept=False)
        cludl = EnCluDL(
            desparsified_lasso=DesparsifiedLasso(estimator=estimator),
            clustering=clustering,
            n_bootstraps=5,
            random_state=seed,
        )
        cludl.fit_importance(X_init, y)
        fdr = 0.1
        selected = cludl.fdr_selection(fdr=fdr, two_tailed_test=False)

        fdp, power = spatially_relaxed_fdp_power(
            selected=selected,
            ground_truth=beta,
            roi_size=roi_size,
            spatial_tolerance=3,
            shape=shape,
        )
        fdp_list.append(fdp)
        power_list.append(power)
    assert np.mean(power_list) >= 0.5
    assert np.mean(fdp_list) <= fdr


def test_cludl_temporal():
    """
    Testing the procedure on two simulations with a 1D data structure and
    with n << p: with a temporal dimension. The support is connected and
    of size 10, it must be recovered with a small spatial tolerance
    parametrized by `margin_size`.
    """
    n_samples, n_features, n_target = 100, 500, 3
    support_size = 10
    signal_noise_ratio = 50.0
    rho_serial = 0.9
    rho_data = 0.9
    n_clusters = 100
    margin_size = 5
    extended_support = support_size + margin_size
    test_tol = 0.05

    fdp_list = []
    power_list = []
    for seed in range(10):
        X, y, beta, noise = multivariate_simulation(
            n_samples=n_samples,
            n_features=n_features,
            n_targets=n_target,
            support_size=support_size,
            signal_noise_ratio=signal_noise_ratio,
            rho_serial=rho_serial,
            rho=rho_data,
            shuffle=False,
            continuous_support=True,
            seed=seed,
        )

        connectivity = image.grid_to_graph(n_x=n_features, n_y=1, n_z=1)
        ward = FeatureAgglomeration(
            n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
        )

        cludl = CluDL(
            desparsified_lasso=DesparsifiedLasso(
                estimator=MultiTaskLassoCV(max_iter=1000)
            ),
            clustering=ward,
            random_state=seed,
        )
        cludl.fit_importance(X, y)

        alpha = 0.05
        selected = cludl.fdr_selection(fdr=alpha, two_tailed_test=False)
        gt_mask = np.zeros(n_features, dtype=int)
        gt_mask[:extended_support] = 1
        fdp, power = fdp_power(
            selected=selected,
            ground_truth=gt_mask,
        )
        fdp_list.append(fdp)
        power_list.append(power)
    assert np.mean(power_list) >= 0.5
    assert np.mean(fdp_list) <= 2 * alpha


def test_encludl_temporal():
    """
    Testing the procedure on two simulations with a 1D data structure and
    with n << p: with a temporal dimension. The support is connected and
    of size 10, it must be recovered with a small spatial tolerance
    parametrized by `margin_size`.
    """
    n_samples, n_features, n_target = 200, 100, 3
    support_size = 10
    signal_noise_ratio = 50.0
    rho_serial = 0.9
    rho_data = 0.5
    n_clusters = 50
    margin_size = 5
    extended_support = support_size + margin_size

    fdp_list = []
    power_list = []
    for seed in range(10):
        X, y, beta, noise = multivariate_simulation(
            n_samples=n_samples,
            n_features=n_features,
            n_targets=n_target,
            support_size=support_size,
            signal_noise_ratio=signal_noise_ratio,
            rho_serial=rho_serial,
            rho=rho_data,
            shuffle=False,
            continuous_support=True,
            seed=seed,
        )

        connectivity = image.grid_to_graph(n_x=n_features, n_y=1, n_z=1)
        ward = FeatureAgglomeration(
            n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
        )

        cludl = EnCluDL(
            desparsified_lasso=DesparsifiedLasso(
                estimator=MultiTaskLassoCV(max_iter=1000)
            ),
            clustering=ward,
            n_bootstraps=5,
            random_state=seed,
        )
        cludl.fit_importance(X, y)

        alpha = 0.1
        selected = cludl.fdr_selection(fdr=alpha, two_tailed_test=False)
        gt_mask = np.zeros(n_features, dtype=int)
        gt_mask[:extended_support] = 1
        fdp, power = fdp_power(
            selected=selected,
            ground_truth=gt_mask,
        )
        fdp_list.append(fdp)
        power_list.append(power)
    assert np.mean(power_list) >= 0.5
    assert np.mean(fdp_list) <= alpha
