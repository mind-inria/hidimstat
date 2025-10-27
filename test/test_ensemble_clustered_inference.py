"""
Test the clustered_inference module
"""

import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_extraction import image
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from hidimstat.ensemble_clustered_inference import EnsembleClusteredInference
from hidimstat.desparsified_lasso import DesparsifiedLasso
from hidimstat._utils.scenario import multivariate_simulation


def set_desparsified_lasso_multi_time():
    multitasklassoCV = MultiTaskLassoCV(
        eps=1e-2,
        fit_intercept=False,
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        tol=1e-4,
        max_iter=5000,
        random_state=1,
        n_jobs=1,
    )
    return DesparsifiedLasso(model_y=multitasklassoCV)


# Scenario 1: data with no temporal dimension
def test_clustered_inference_no_temporal():
    """
    Testing the procedure on one simulations with a 1D data structure and
    with n << p: no temporal dimension. The support is connected and of
    size 10, it must be recovered with a small spatial tolerance
    parametrized by `margin_size`.
    Computing one sided p-values, we want low p-values for the features of
    the support and p-values close to 0.5 for the others.
    """

    n_samples, n_features = 100, 2000
    support_size = 10
    signal_noise_ratio = 5.0
    rho = 0.95
    n_clusters = 150
    margin_size = 5
    interior_support = support_size - margin_size
    extended_support = support_size + margin_size

    X_init, y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        signal_noise_ratio=signal_noise_ratio,
        rho=rho,
        shuffle=False,
        continuous_support=True,
        seed=2,
    )

    y = y - np.mean(y)
    X_init = X_init - np.mean(X_init, axis=0)

    connectivity = image.grid_to_graph(n_x=n_features, n_y=1, n_z=1)
    ward = FeatureAgglomeration(
        n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
    )

    clustered_inference = EnsembleClusteredInference(
        ward=ward, scaler_sampling=StandardScaler(), n_bootstraps=1
    ).fit(X_init, y)
    clustered_inference.importance(X_init, y)

    expected = 0.5 * np.ones(n_features)
    expected[:support_size] = 0.0

    assert_almost_equal(
        clustered_inference.pvalues_corr_[:interior_support],
        expected[:interior_support],
    )
    assert_almost_equal(
        clustered_inference.pvalues_corr_[extended_support:200],
        expected[extended_support:200],
        decimal=1,
    )


# Scenario 2: temporal data
def test_clustered_inference_temporal():
    """
    Testing the procedure on two simulations with a 1D data structure and
    with n << p: with a temporal dimension. The support is connected and
    of size 10, it must be recovered with a small spatial tolerance
    parametrized by `margin_size`.
    Computing one sided p-values, we want low p-values for the features of
    the support and p-values close to 0.5 for the others.
    """
    n_samples, n_features, n_target = 200, 2000, 10
    support_size = 10
    signal_noise_ratio = 50.0
    rho_serial = 0.9
    rho_data = 0.9
    n_clusters = 150
    margin_size = 5
    interior_support = support_size - margin_size
    extended_support = support_size + margin_size

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
        seed=10,
    )

    connectivity = image.grid_to_graph(n_x=n_features, n_y=1, n_z=1)
    ward = FeatureAgglomeration(
        n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
    )

    clustered_inference = EnsembleClusteredInference(
        ward=ward,
        variable_importance=set_desparsified_lasso_multi_time(),
        scaler_sampling=StandardScaler(),
        n_bootstraps=1,
    ).fit(X, y)
    clustered_inference.importance(X, y)

    expected = 0.5 * np.ones(n_features)
    expected[:support_size] = 0.0

    assert_almost_equal(
        clustered_inference.pvalues_corr_[:interior_support],
        expected[:interior_support],
        decimal=3,
    )
    assert_almost_equal(
        clustered_inference.pvalues_corr_[extended_support:],
        expected[extended_support:],
        decimal=1,
    )


# Scenario 3: data with no temporal dimension and with groups
def test_clustered_inference_no_temporal_groups():
    """
    Testing the procedure on one simulations with a 1D data structure and
    with n << p: no temporal dimension. The support is connected and of
    size 10, it must be recovered with a small spatial tolerance
    parametrized by `margin_size`.
    We group the sample in 10 groups of size 10.
    Computing one sided p-values, we want low p-values for the features of
    the support and p-values close to 0.5 for the others.
    """

    n_samples, n_features = 20, 1500
    support_size = 10
    n_groups = 10
    signal_noise_ratio = 5.0
    rho = 0.95
    n_clusters = 150
    margin_size = 5
    interior_support = support_size - margin_size
    extended_support = support_size + margin_size

    # create n_group of samples
    X_ = []
    y_ = []
    for i in range(n_groups):
        X_init, y, beta, noise = multivariate_simulation(
            n_samples=n_samples,
            n_features=n_features,
            support_size=support_size,
            signal_noise_ratio=signal_noise_ratio,
            rho=rho,
            shuffle=False,
            continuous_support=True,
            seed=4 + i,
        )
        X_.append(X_init)
        y_.append(y)

    y_ = np.concatenate(y_)
    y_ = y_ - np.mean(y_)
    X_ = np.concatenate(X_)
    X_ = X_ - np.mean(X_, axis=0)
    groups = np.repeat(np.arange(0, n_groups), n_samples)

    connectivity = image.grid_to_graph(n_x=n_features, n_y=1, n_z=1)
    ward = FeatureAgglomeration(
        n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
    )

    clustered_inference = EnsembleClusteredInference(
        ward=ward,
        scaler_sampling=StandardScaler(),
        groups=groups,
        n_bootstraps=1,
    ).fit(X_, y_)
    clustered_inference.importance(X_, y_)

    expected = 0.5 * np.ones(n_features)
    expected[:support_size] = 0.0

    assert_almost_equal(
        clustered_inference.pvalues_corr_[:interior_support],
        expected[:interior_support],
    )
    assert_almost_equal(
        clustered_inference.pvalues_corr_[extended_support:200],
        expected[extended_support:200],
        decimal=1,
    )


def test_ensemble_clustered_inference():
    """Testing the procedure on a simulation with a 1D data structure
    and with n << p: the first test has no temporal dimension, the second has a
    temporal dimension. The support is connected and of size 10, it must be
    recovered with a small spatial tolerance parametrized by `margin_size`.
    Computing one sided p-values, we want low p-values for the features of
    the support and p-values close to 0.5 for the others."""

    # Scenario 1: data with no temporal dimension
    # ###########################################
    n_samples, n_features = 200, 2000
    support_size = 10
    signal_noise_ratio = 5.0
    rho = 0.95

    X_init, y, beta, noise = multivariate_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        signal_noise_ratio=signal_noise_ratio,
        rho=rho,
        shuffle=False,
        continuous_support=True,
        seed=0,
    )

    margin_size = 5
    n_clusters = 200
    n_bootstraps = 3

    y = y - np.mean(y)
    X_init = X_init - np.mean(X_init, axis=0)

    connectivity = image.grid_to_graph(n_x=n_features, n_y=1, n_z=1)
    ward = FeatureAgglomeration(
        n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
    )

    EnCluDl = EnsembleClusteredInference(
        ward=ward,
        scaler_sampling=StandardScaler(),
        n_bootstraps=n_bootstraps,
    ).fit(X_init, y)
    EnCluDl.importance(X_init, y)
    selected = EnCluDl.fdr_selection(fdr=0.1)

    expected = np.zeros(n_features)
    expected[:support_size] = 1.0

    assert_almost_equal(
        selected[: support_size - margin_size], expected[: support_size - margin_size]
    )
    assert_almost_equal(
        selected[support_size + margin_size :],
        expected[support_size + margin_size :],
        decimal=1,
    )


def test_ensemble_clustered_inference_temporal_data():
    "Test with temporal data"
    # Scenario 2: temporal data
    # #########################
    n_samples, n_features, n_target = 200, 400, 10
    support_size = 10
    signal_noise_ratio = 5.0
    rho_serial = 0.9
    rho_data = 0.9
    n_clusters = 50
    margin_size = 5
    interior_support = support_size - margin_size
    extended_support = support_size + margin_size
    n_bootstraps = 4

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
        seed=7,
    )

    connectivity = image.grid_to_graph(n_x=n_features, n_y=1, n_z=1)
    ward = FeatureAgglomeration(
        n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
    )

    EnCluDl = EnsembleClusteredInference(
        variable_importance=set_desparsified_lasso_multi_time(),
        ward=ward,
        scaler_sampling=StandardScaler(),
        n_bootstraps=n_bootstraps,
    ).fit(X, y)
    EnCluDl.importance(X, y)
    selected = EnCluDl.fdr_selection(fdr=0.1, fdr_control="bhq")

    expected = np.zeros(n_features)
    expected[:support_size] = 1.0

    assert_almost_equal(
        selected[:interior_support], expected[:interior_support], decimal=3
    )
    assert_almost_equal(
        selected[extended_support:], expected[extended_support:], decimal=1
    )

    # different aggregation method
    selected = EnCluDl.fdr_selection(fdr=0.1, fdr_control="bhy")

    expected = np.zeros(n_features)
    expected[:support_size] = 1.0

    assert_almost_equal(
        selected[:interior_support], expected[:interior_support], decimal=3
    )
    assert_almost_equal(
        selected[extended_support:], expected[extended_support:], decimal=1
    )
