"""
Test the clustered_inference module
"""

import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_extraction import image

from hidimstat.clustered_inference import clustered_inference
from hidimstat.scenario import (
    multivariate_1D_simulation,
    multivariate_temporal_simulation,
)


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
    n_clusters = 200
    margin_size = 5
    interior_support = support_size - margin_size
    extended_support = support_size + margin_size

    X_init, y, beta, epsilon = multivariate_1D_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        sigma=sigma,
        rho=rho,
        shuffle=False,
        seed=2,
    )

    y = y - np.mean(y)
    X_init = X_init - np.mean(X_init, axis=0)

    connectivity = image.grid_to_graph(n_x=n_features, n_y=1, n_z=1)
    ward = FeatureAgglomeration(
        n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
    )

    beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = (
        clustered_inference(X_init, y, ward, n_clusters)
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
    n_clusters = 200
    margin_size = 5
    interior_support = support_size - margin_size
    extended_support = support_size + margin_size

    X, Y, beta, noise = multivariate_temporal_simulation(
        n_samples=n_samples,
        n_features=n_features,
        n_times=n_times,
        support_size=support_size,
        sigma=sigma,
        rho_noise=rho_noise,
        rho_data=rho_data,
        shuffle=False,
    )

    connectivity = image.grid_to_graph(n_x=n_features, n_y=1, n_z=1)
    ward = FeatureAgglomeration(
        n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
    )

    beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = (
        clustered_inference(X, Y, ward, n_clusters, method="desparsified-group-lasso")
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
        X_init, y, beta, epsilon = multivariate_1D_simulation(
            n_samples=n_samples,
            n_features=n_features,
            support_size=support_size,
            sigma=sigma,
            rho=rho,
            shuffle=False,
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

    beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = (
        clustered_inference(X_, y_, ward, n_clusters, groups=groups)
    )

    expected = 0.5 * np.ones(n_features)
    expected[:support_size] = 0.0

    assert_almost_equal(pval_corr[:interior_support], expected[:interior_support])
    assert_almost_equal(
        pval_corr[extended_support:200], expected[extended_support:200], decimal=1
    )


# Scenario 1: data with no temporal dimension
def test_clustered_inference_exception_methods():
    """
    Testing the procedure on two simulations with a 1D data structure and
    checking that the procedure raises an exception when an unknown method is
    provided.
    """
    n_samples, n_features = 100, 2000
    n_clusters = 200

    X_init, y, beta, epsilon = multivariate_1D_simulation(
        n_samples=n_samples,
        n_features=n_features,
        shuffle=False,
        seed=2,
    )

    y = y - np.mean(y)
    X_init = X_init - np.mean(X_init, axis=0)

    connectivity = image.grid_to_graph(n_x=n_features, n_y=1, n_z=1)
    ward = FeatureAgglomeration(
        n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
    )

    with pytest.raises(ValueError, match="Unknow method"):
        clustered_inference(X_init, y, ward, n_clusters, method="lll")
