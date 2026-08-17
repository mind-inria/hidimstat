"""
Test the clustered_inference module
"""

import numpy as np
import pytest
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_extraction import image
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
from sklearn.model_selection import train_test_split

from hidimstat import ClusterImportance, DesparsifiedLasso, EnsembleImportance
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


def test_ensemble_parameter_check():
    """
    Test to verify parameter's class compliance.
    """
    en_vi = EnsembleImportance(vim=LassoCV())
    with pytest.raises(
        AssertionError,
        match="estimator needs to be a subclass of BaseVariableImportance",
    ):
        en_vi.fit(np.zeros((5, 5)), np.zeros((5,)))


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(100, 20, 10, 0.5, 42, 1.0, 50.0, 0.9)],
    ids=["basic data"],
)
def test_ensemble_importance_check_fit(data_generator):
    """
    Check that a call to importance() fails if EnsembleImportance is not fitted.
    """
    X, y, _ = data_generator

    encludl = EnsembleImportance(
        vim=DesparsifiedLasso(estimator=LassoCV()),
        n_repeats=5,
        random_state=42,
    )

    with pytest.raises(
        ValueError, match="The estimators need to be fit before using them"
    ):
        encludl.importance(X, y)


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [(150, 200, 10, 0, 42, 1.0, 10.0, 0.0)],
    ids=["basic data"],
)
def test_ensemble_importance(data_generator):
    """Test the EnsembleImportance algorithm on a linear scenario."""
    X, y, important_mask = data_generator
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    dl = DesparsifiedLasso(estimator=LassoCV())
    dl.fit(X_train, y_train)

    endl = EnsembleImportance(
        vim=dl,
        n_repeats=5,
        random_state=0,
    )
    endl.fit(
        X_train,
        y_train,
    )
    importance = endl.importance(X_test, y_test)

    assert importance.shape == (X.shape[1],)
    assert (
        importance[important_mask].mean() > importance[~important_mask].mean()
    )


def test_encluvi_spatial():
    """
    Test EnCluVI on a 2D spatial simulation. Testing for support recovery methods using
    clustering is challenging as clusters that intersect the true support can also
    include non-support features, rapidly increasing false positives. To address this,
    we introduce a spatial relaxation in the evaluation metrics.

     - Test that the spatially relaxed FDP is below a specified FDR threshold (0.1).
     - Test that the statistical power is above a specified threshold (0.8).
    """
    n_samples = 400
    shape = (10, 10)
    n_features = shape[1] * shape[0]
    roi_size = 2  # size of the edge of the four predictive regions
    signal_noise_ratio = 32.0  # noise standard deviation
    smooth_X = (
        0.2  # level of spatial smoothing introduced by the Gaussian filter
    )
    tol = 0.1

    fp_list = []
    power_list = []
    for seed in range(10):
        # generating the data
        X_init, y, beta, _ = multivariate_simulation_spatial(
            n_samples, shape, roi_size, signal_noise_ratio, smooth_X, seed=seed
        )

        y = y - np.mean(y)
        X_init = X_init - np.mean(X_init, axis=0)

        n_clusters = 50
        connectivity = image.grid_to_graph(n_x=n_features, n_y=1, n_z=1)
        clustering = FeatureAgglomeration(
            n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
        )

        estimator = LassoCV(
            max_iter=1000, tol=0.0001, eps=0.01, fit_intercept=False
        )
        encludl = EnsembleImportance(
            vim=ClusterImportance(
                vim=DesparsifiedLasso(estimator=estimator),
                clustering=clustering,
            ),
            n_repeats=5,
            random_state=seed,
        )
        encludl.fit_importance(X_init, y)
        fwer = 0.1
        selected = encludl.fwer_selection(fwer=fwer, two_tailed_test=False)

        fdp, power = spatially_relaxed_fdp_power(
            selected=selected,
            ground_truth=beta,
            roi_size=roi_size,
            spatial_tolerance=2,
            shape=shape,
        )
        fp_list.append(int(fdp > 0))
        power_list.append(power)

    assert np.mean(power_list) >= 0.5
    assert np.mean(fp_list) <= fwer + tol


def test_encluvi_temporal():
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
    rho_data = 0.5
    n_clusters = 50
    margin_size = 5
    extended_support = support_size + margin_size

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

        encludl = EnsembleImportance(
            vim=ClusterImportance(
                vim=DesparsifiedLasso(
                    estimator=MultiTaskLassoCV(max_iter=1000)
                ),
                clustering=ward,
            ),
            n_repeats=5,
            random_state=seed,
        )
        encludl.fit_importance(X, y)

        alpha = 0.1
        selected = encludl.fdr_selection(fdr=alpha, two_tailed_test=False)
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


def test_encluvi_independence():
    """Test that EnCluVI works with n_jobs=1. Non regression for #425"""
    n_samples = 50
    shape = (20, 20)
    roi_size = 4  # size of the edge of the four predictive regions
    X_init, y, _, _ = multivariate_simulation_spatial(
        n_samples, shape, roi_size, signal_noise_ratio=10.0, smooth_X=1
    )
    alpha = 0.05  # alpha is the significance level for the statistical test
    n_clusters = 50
    connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1])
    ward = FeatureAgglomeration(
        n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
    )

    encluvi = EnsembleImportance(
        vim=ClusterImportance(
            vim=DesparsifiedLasso(estimator=LassoCV()),
            clustering=ward,
        ),
        bootstrap_frac=0.7,
        n_repeats=20,
        random_state=1,
        n_jobs=1,
    )
    encluvi.fit_importance(X_init, y)
    selected_ecdl = encluvi.fwer_selection(alpha, n_tests=n_clusters)
    assert np.sum(selected_ecdl) > 10


@pytest.fixture(scope="module")
def ensemble_test_data():
    """
    Fixture to generate test data and a fitted LinearRegression model for CFI
    reproducibility tests.
    """
    X, y, _, _ = multivariate_simulation(
        n_samples=100,
        n_features=5,
        support_size=2,
        rho=0,
        value=1,
        signal_noise_ratio=4,
        rho_serial=0,
        shuffle=False,
        seed=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    dl = DesparsifiedLasso(estimator=LassoCV())
    dl.fit(X_train, y_train)
    en_dl_default_parameters = {
        "vim": dl,
        "n_repeats": 10,
        "bootstrap_frac": 0.5,
        "n_jobs": 1,
    }
    return X_train, y_train, X_test, y_test, en_dl_default_parameters


def test_ensemble_repeatibility(ensemble_test_data):
    """
    Test that multiple calls of .importance() when EnsembleImportance is not seeded
    provides different results.
    """
    X_train, y_train, X_test, y_test, ensemble_default_parameters = (
        ensemble_test_data
    )
    endl = EnsembleImportance(**ensemble_default_parameters)
    endl.fit(X_train, y_train)
    importance = endl.importance(X_test, y_test)
    # repeat
    endl.fit(X_train, y_train)
    importance_repeat = endl.importance(X_test, y_test)
    assert not np.array_equal(importance, importance_repeat)


def test_ensemble_randomness_with_none(ensemble_test_data):
    """
    Test randomness of multiple calls of .importance() when
    EnsembleImportance has random_state=None
    """
    X_train, y_train, X_test, y_test, ensemble_default_parameters = (
        ensemble_test_data
    )
    endl = EnsembleImportance(random_state=None, **ensemble_default_parameters)
    endl.fit(X_train, y_train)
    importance = endl.importance(X_test, y_test)

    # reproducibility
    endl2 = EnsembleImportance(
        random_state=None, **ensemble_default_parameters
    )
    endl2.fit(X_train, y_train)
    importance_reproducibility = endl2.importance(X_test, y_test)

    assert not np.array_equal(importance, importance_reproducibility)


def test_ensemble_randomness_with_integer(ensemble_test_data):
    """
    Test reproducibility of multiple calls of .importance() when
    EnsembleImportance has random_state=42
    """
    X_train, y_train, X_test, y_test, ensemble_default_parameters = (
        ensemble_test_data
    )
    endl = EnsembleImportance(random_state=42, **ensemble_default_parameters)
    endl.fit(X_train, y_train)
    importance = endl.importance(X_test, y_test)

    # refit
    endl.fit(X_train, y_train)
    importance_refit = endl.importance(X_test, y_test)
    assert np.array_equal(importance, importance_refit)

    # reproducibility
    endl2 = EnsembleImportance(random_state=42, **ensemble_default_parameters)
    endl2.fit(X_train, y_train)
    importance_reproducibility = endl2.importance(X_test, y_test)
    assert np.array_equal(importance, importance_reproducibility)


def test_ensemble_randomness_with_rng(ensemble_test_data):
    """
    Test that:
     1. Multiple calls of .importance() when EnsembleImportance has random_state=rng are random
     2. refit with same rng provides same result
    """
    X_train, y_train, X_test, y_test, ensemble_default_parameters = (
        ensemble_test_data
    )
    rng = np.random.default_rng(42)
    endl = EnsembleImportance(random_state=rng, **ensemble_default_parameters)
    endl.fit(X_train, y_train)
    importance = endl.importance(X_test, y_test)

    # refit
    endl.fit(X_train, y_train)
    importance_refit = endl.importance(X_test, y_test)
    assert not np.array_equal(importance, importance_refit)

    # refit repeatability
    rng = np.random.default_rng(42)
    endl.random_state = rng
    endl.fit(X_train, y_train)
    importance_refit2 = endl.importance(X_test, y_test)
    assert np.array_equal(importance, importance_refit2)

    # reproducibility
    endl2 = EnsembleImportance(
        random_state=np.random.default_rng(42), **ensemble_default_parameters
    )
    endl2.fit(X_train, y_train)
    importance_reproducibility = endl2.importance(X_test, y_test)
    assert np.array_equal(importance, importance_reproducibility)
