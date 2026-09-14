"""
Test the clustered_inference module
"""

import numpy as np
import pytest
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_extraction import image
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
from sklearn.model_selection import train_test_split

from hidimstat import ClusterImportance, DesparsifiedLasso
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
    roi_size_extended = roi_size + spatial_tolerance
    ground_truth_extended = ground_truth.copy().reshape(shape)
    ground_truth_extended[0:roi_size_extended, 0:roi_size_extended] += 1
    ground_truth_extended[-roi_size_extended:, -roi_size_extended:] += 1
    ground_truth_extended[0:roi_size_extended, -roi_size_extended:] += 1
    ground_truth_extended[-roi_size_extended:, 0:roi_size_extended] += 1
    ground_truth_extended = (ground_truth_extended > 0).astype(bool).flatten()

    true_positive = np.sum(selected.astype(bool) & ground_truth.astype(bool))
    false_positive = np.sum(selected.astype(bool) & ~ground_truth_extended)

    fdp = false_positive / np.sum(selected)
    power = true_positive / np.sum(ground_truth)
    return fdp, power


def test_cluster_parameter_check():
    """
    Test to verify parameter's class compliance.
    """
    clu_vi = ClusterImportance(
        vim=LassoCV(),
        clustering=FeatureAgglomeration(),
    )
    with pytest.raises(
        AssertionError,
        match="estimator needs to be a subclass of BaseVariableImportance",
    ):
        clu_vi.fit(np.zeros((5, 5)), np.zeros((5,)))

    clu_vi = ClusterImportance(
        vim=DesparsifiedLasso(estimator=LassoCV()),
        clustering=LassoCV(),
    )
    with pytest.raises(
        AssertionError,
        match=r"clustering needs to be an instance of sklearn\.cluster\.FeatureAgglomeration",
    ):
        clu_vi.fit(np.zeros((5, 5)), np.zeros((5,)))


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(100, 20, 10, 0.5, 42, 1.0, 50.0, 0.9)],
    ids=["basic corr data"],
)
def test_cluster_importance_check_fit(data_generator):
    """
    Check that a call to importance() fails if ClusterImportance is not fitted.
    """
    X, y, _ = data_generator

    cludl = ClusterImportance(
        vim=DesparsifiedLasso(estimator=LassoCV()),
        clustering=FeatureAgglomeration(),
    )

    with pytest.raises(
        ValueError, match="The estimator needs to be fit before using them"
    ):
        cludl.importance(X, y)


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(150, 20, 4, 0, 42, 1.0, 10.0, 0.0)],
    ids=["basic data"],
)
def test_cluster_importance(data_generator):
    """Test the ClusterImportance algorithm on a linear scenario."""
    X, y, beta = data_generator

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    dl = DesparsifiedLasso(estimator=LassoCV())
    dl.fit(X_train, y_train)

    importance_stack = np.zeros((0, X.shape[1]))
    n_repeats = 10

    for _ in range(n_repeats):
        cludl = ClusterImportance(vim=dl, clustering=FeatureAgglomeration())

        cludl.fit(
            X_train,
            y_train,
        )
        importance = cludl.importance(X_test, y_test)
        assert importance.shape == (X.shape[1],)
        importance_stack = np.vstack((importance_stack, importance))

    importance = importance_stack.mean(axis=0)

    assert importance[beta].mean() > importance[~beta].mean()


def test_cluvi_spatial():
    """
    Test ClusterImportance on a 2D spatial simulation. Testing for support recovery methods using
    clustering is challenging as clusters that intersect the true support can also
    include non-support features, rapidly increasing false positives. To address this,
    we introduce a spatial relaxation in the evaluation metrics.

     - Test that the spatially relaxed FDP is below a specified FDR threshold (0.1).
     - Test that the statistical power is above a specified threshold (0.8).
    """
    n_samples = 50
    shape = (10, 10)
    n_features = shape[1] * shape[0]
    roi_size = 2  # size of the edge of the four predictive regions
    signal_noise_ratio = 32.0  # noise standard deviation
    smooth_X = (
        0.2  # level of spatial smoothing introduced by the Gaussian filter
    )

    fp_list = []
    power_list = []
    for seed in range(10):
        # generating the data
        X_init, y, beta, _ = multivariate_simulation_spatial(
            n_samples, shape, roi_size, signal_noise_ratio, smooth_X, seed=seed
        )

        y = y - np.mean(y)
        X_init = X_init - np.mean(X_init, axis=0)

        n_clusters = 20
        connectivity = image.grid_to_graph(n_x=n_features, n_y=1, n_z=1)
        clustering = FeatureAgglomeration(
            n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
        )
        estimator = LassoCV(
            max_iter=1000, tol=0.0001, eps=0.01, fit_intercept=False
        )
        cluvi = ClusterImportance(
            vim=DesparsifiedLasso(estimator=estimator),
            clustering=clustering,
        )
        cluvi.fit_importance(X_init, y)
        fwer = 0.1
        selected = cluvi.fwer_selection(fwer=fwer, two_tailed_test=False)

        fdp, power = spatially_relaxed_fdp_power(
            selected=selected,
            ground_truth=beta,
            roi_size=roi_size,
            spatial_tolerance=3,
            shape=shape,
        )
        fp_list.append(int(fdp > 0))
        power_list.append(power)

    assert np.mean(power_list) >= 0.5
    assert np.mean(fp_list) <= fwer


def test_cluvi_independence():
    """Test that CluVI works with repeated calls
    non-regression test for #425
    """
    n_samples = 50
    shape = (20, 20)
    roi_size = 4  # size of the edge of the four predictive regions
    X_init, y, _, _ = multivariate_simulation_spatial(
        n_samples, shape, roi_size, signal_noise_ratio=10.0, smooth_X=1
    )
    alpha = 0.05  # alpha is the significance level for the statistical test
    n_clusters = 20
    connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1])
    ward = FeatureAgglomeration(
        n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
    )
    vim = DesparsifiedLasso(
        estimator=LassoCV(
            max_iter=1000, tol=0.0001, eps=0.01, fit_intercept=False
        )
    )

    c1 = ClusterImportance(
        vim=vim,
        clustering=ward,
    )
    c1.fit_importance(X_init, y)
    s1 = c1.fwer_selection(alpha, n_tests=n_clusters)
    s2_iterations = np.zeros((0, len(s1)))
    n_iterations = 20
    for _ in range(n_iterations):
        c2 = ClusterImportance(vim=vim, clustering=ward)
        c2.fit_importance(X_init, y)
        s2 = c2.fwer_selection(alpha, n_tests=n_clusters)
        s2_iterations = np.vstack((s2_iterations, s2))

    assert np.sum(s1) != 0
    assert (
        np.abs(np.sum(s2_iterations) / n_iterations - np.sum(s1)) / np.sum(s1)
        < 0.5
    )


def test_cluvi_temporal():
    """
    Testing the procedure on two simulations with a 1D data structure and
    with n << p: with a temporal dimension. The support is connected and
    of size 10, it must be recovered with a small spatial tolerance
    parametrized by `margin_size`.
    """
    n_samples, n_features, n_target = 50, 200, 3
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
        X, y, _, _ = multivariate_simulation(
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

        cluvi = ClusterImportance(
            vim=DesparsifiedLasso(estimator=MultiTaskLassoCV(max_iter=1000)),
            clustering=ward,
        )
        cluvi.fit_importance(X, y)

        alpha = 0.05
        selected = cluvi.fdr_selection(fdr=alpha)
        gt_mask = np.zeros(n_features, dtype=int)
        gt_mask[:extended_support] = 1
        fdp, power = fdp_power(
            selected=selected,
            ground_truth=gt_mask,
        )
        fdp_list.append(fdp)
        power_list.append(power)

    assert np.mean(power_list) >= 0.5 - test_tol
    assert np.mean(fdp_list) <= alpha + test_tol
