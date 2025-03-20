"""
Test the ensemble_clustered_inference module
"""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_extraction import image

from hidimstat.ensemble_clustered_inference import ensemble_clustered_inference
from hidimstat.scenario import (
    multivariate_1D_simulation,
    multivariate_temporal_simulation,
)


def test_ensemble_clustered_inference_senario_1():
    """Testing the procedure on a simulation with a 1D data structure
    and with n << p: the first test has no temporal dimension, the second has a
    temporal dimension. The support is connected and of size 10, it must be
    recovered with a small spatial tolerance parametrized by `margin_size`.
    Computing one sided p-values, we want low p-values for the features of
    the support and p-values close to 0.5 for the others."""

    # Scenario 1: data with no temporal dimension
    # ###########################################
    inference_method = "desparsified-lasso"
    n_samples, n_features = 100, 2000
    support_size = 10
    sigma = 5.0
    rho = 0.95

    X_init, y, beta, epsilon = multivariate_1D_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        sigma=sigma,
        rho=rho,
        shuffle=False,
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

    beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = (
        ensemble_clustered_inference(
            X_init,
            y,
            ward,
            n_clusters,
            n_bootstraps=n_bootstraps,
            inference_method=inference_method,
        )
    )

    expected = 0.5 * np.ones(n_features)
    expected[:support_size] = 0.0

    assert_almost_equal(
        pval_corr[: support_size - margin_size], expected[: support_size - margin_size]
    )
    assert_almost_equal(
        pval_corr[support_size + margin_size :],
        expected[support_size + margin_size :],
        decimal=1,
    )

def test_ensemble_clustered_inference_senario_2():
    # Scenario 2: temporal data
    # #########################
    inference_method = "desparsified-group-lasso"
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
        ensemble_clustered_inference(
            X,
            Y,
            ward,
            n_clusters,
            n_bootstraps=n_bootstraps,
            inference_method=inference_method,
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

    beta_hat, pval, pval_corr, one_minus_pval, one_minus_pval_corr = (
        ensemble_clustered_inference(
            X,
            Y,
            ward,
            n_clusters,
            n_bootstraps=n_bootstraps,
            inference_method=inference_method,
            ensembling_method="medians",
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


def test_ensemble_clustered_inference_exception():
    """
    Test the raise of exception
    """
    n_samples, n_features = 100, 2000
    n_clusters = 10
    X, Y, beta, epsilon = multivariate_1D_simulation(
        n_samples=n_samples,
        n_features=n_features,
    )
    connectivity = image.grid_to_graph(n_x=n_features, n_y=1, n_z=1)
    ward = FeatureAgglomeration(
        n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
    )

    # Test the raise of exception
    with pytest.raises(ValueError, match="Unknown ensembling method."):
        ensemble_clustered_inference(
            X, Y, ward, n_clusters, ensembling_method="wrong_method"
        )

    with pytest.raises(
        ValueError,
        match="'memory' must be None or a string corresponding "
        + "to the path of the caching directory.",
    ):
        ensemble_clustered_inference(X, Y, ward, n_clusters, memory=[])
