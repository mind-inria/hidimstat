"""
Test the clustered_inference module
"""

import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import image

from hidimstat.ensemble_clustered_inference import (
    clustered_inference,
    clustered_inference_pvalue,
)
from hidimstat.ensemble_clustered_inference import (
    ensemble_clustered_inference,
    ensemble_clustered_inference_pvalue,
)
from hidimstat._utils.scenario import multivariate_simulation_autoregressive


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
    sigma = 5.0
    rho = 0.95
    n_clusters = 150
    margin_size = 5
    interior_support = support_size - margin_size
    extended_support = support_size + margin_size

    X_init, y, beta, _, _, _ = multivariate_simulation_autoregressive(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        sigma_noise=sigma,
        rho=rho,
        shuffle=False,
        continue_support=True,
        seed=2,
    )

    y = y - np.mean(y)
    X_init = X_init - np.mean(X_init, axis=0)

    connectivity = image.grid_to_graph(n_x=n_features, n_y=1, n_z=1)
    ward = FeatureAgglomeration(
        n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
    )

    ward_, beta_hat, theta_hat, precision_diag = clustered_inference(
        X_init, y, ward, n_clusters, scaler_sampling=StandardScaler()
    )

    beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = (
        clustered_inference_pvalue(
            n_samples, None, ward_, beta_hat, theta_hat, precision_diag
        )
    )

    expected = 0.5 * np.ones(n_features)
    expected[:support_size] = 0.0

    assert_almost_equal(pval_corr[:interior_support], expected[:interior_support])
    assert_almost_equal(
        pval_corr[extended_support:200], expected[extended_support:200], decimal=1
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
    n_samples, n_features, n_times = 200, 2000, 10
    support_size = 10
    sigma = 5.0
    rho_noise = 0.9
    rho_data = 0.9
    n_clusters = 150
    margin_size = 5
    interior_support = support_size - margin_size
    extended_support = support_size + margin_size

    X, y, beta, _, _, _ = multivariate_simulation_autoregressive(
        n_samples=n_samples,
        n_features=n_features,
        n_times=n_times,
        support_size=support_size,
        sigma_noise=sigma,
        rho_noise_time=rho_noise,
        rho=rho_data,
        shuffle=False,
        continue_support=True,
    )

    connectivity = image.grid_to_graph(n_x=n_features, n_y=1, n_z=1)
    ward = FeatureAgglomeration(
        n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
    )

    ward_, beta_hat, theta_hat, precision_diag = clustered_inference(
        X, y, ward, n_clusters, scaler_sampling=StandardScaler()
    )

    beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = (
        clustered_inference_pvalue(
            n_samples, True, ward_, beta_hat, theta_hat, precision_diag
        )
    )

    expected = 0.5 * np.ones(n_features)
    expected[:support_size] = 0.0

    assert_almost_equal(
        pval_corr[:interior_support], expected[:interior_support], decimal=3
    )
    assert_almost_equal(
        pval_corr[extended_support:], expected[extended_support:], decimal=1
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

    n_samples, n_features = 20, 2000
    support_size = 10
    n_groups = 10
    sigma = 5.0
    rho = 0.95
    n_clusters = 200
    margin_size = 5
    interior_support = support_size - margin_size
    extended_support = support_size + margin_size

    # create n_group of samples
    X_ = []
    y_ = []
    for i in range(n_groups):
        X_init, y, beta, _, _, _ = multivariate_simulation_autoregressive(
            n_samples=n_samples,
            n_features=n_features,
            support_size=support_size,
            sigma_noise=sigma,
            rho=rho,
            shuffle=False,
            continue_support=True,
            seed=2 + i,
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

    ward_, beta_hat, theta_hat, precision_diag = clustered_inference(
        X_, y_, ward, n_clusters, groups=groups, scaler_sampling=StandardScaler()
    )

    beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = (
        clustered_inference_pvalue(
            n_groups * n_samples, False, ward_, beta_hat, theta_hat, precision_diag
        )
    )

    expected = 0.5 * np.ones(n_features)
    expected[:support_size] = 0.0

    assert_almost_equal(pval_corr[:interior_support], expected[:interior_support])
    assert_almost_equal(
        pval_corr[extended_support:200], expected[extended_support:200], decimal=1
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
    sigma = 5.0
    rho = 0.95

    X_init, y, beta, _, _, _ = multivariate_simulation_autoregressive(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        sigma_noise=sigma,
        rho=rho,
        shuffle=False,
        continue_support=True,
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

    list_ward, list_beta_hat, list_theta_hat, list_precision_diag = (
        ensemble_clustered_inference(
            X_init,
            y,
            ward,
            n_clusters,
            scaler_sampling=StandardScaler(),
            n_bootstraps=n_bootstraps,
        )
    )
    beta_hat, selected = ensemble_clustered_inference_pvalue(
        n_samples,
        False,
        list_ward,
        list_beta_hat,
        list_theta_hat,
        list_precision_diag,
    )

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
    n_samples, n_features, n_times = 200, 2000, 10
    support_size = 10
    sigma = 5.0
    rho_noise = 0.9
    rho_data = 0.9
    n_clusters = 200
    margin_size = 5
    interior_support = support_size - margin_size
    extended_support = support_size + margin_size
    n_bootstraps = 4

    X, y, beta, _, _, _ = multivariate_simulation_autoregressive(
        n_samples=n_samples,
        n_features=n_features,
        n_times=n_times,
        support_size=support_size,
        sigma_noise=sigma,
        rho_noise_time=rho_noise,
        rho=rho_data,
        shuffle=False,
        continue_support=True,
    )

    connectivity = image.grid_to_graph(n_x=n_features, n_y=1, n_z=1)
    ward = FeatureAgglomeration(
        n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
    )

    list_ward, list_beta_hat, list_theta_hat, list_precision_diag = (
        ensemble_clustered_inference(
            X,
            y,
            ward,
            n_clusters,
            scaler_sampling=StandardScaler(),
            n_bootstraps=n_bootstraps,
        )
    )
    beta_hat, selected = ensemble_clustered_inference_pvalue(
        n_samples,
        True,
        list_ward,
        list_beta_hat,
        list_theta_hat,
        list_precision_diag,
        fdr_control="bhq",
    )

    expected = np.zeros(n_features)
    expected[:support_size] = 1.0

    assert_almost_equal(
        selected[:interior_support, 0], expected[:interior_support], decimal=3
    )
    assert_almost_equal(
        selected[extended_support:, 0], expected[extended_support:], decimal=1
    )

    # different aggregation method
    beta_hat, selected = ensemble_clustered_inference_pvalue(
        n_samples,
        True,
        list_ward,
        list_beta_hat,
        list_theta_hat,
        list_precision_diag,
        fdr_control="bhy",
    )

    expected = np.zeros(n_features)
    expected[:support_size] = 1.0

    assert_almost_equal(
        selected[:interior_support, 0], expected[:interior_support], decimal=3
    )
    assert_almost_equal(
        selected[extended_support:, 0], expected[extended_support:], decimal=1
    )
