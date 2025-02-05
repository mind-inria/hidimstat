"""
Test the permutation test module
"""

import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.svm import LinearSVR
from sklearn.base import clone

from hidimstat.permutation_test import permutation_test, permutation_test_pval
from hidimstat.scenario import multivariate_1D_simulation


def test_permutation_test():
    """
    Testing the procedure on a simulation with no structure and a support
    of size 1. Computing one-sided p-values, we want a low p-value
    for the first feature and p-values close to 0.5 for the others.
    """

    # parameters for generating the data
    n_samples, n_features = 20, 50
    n_permutations = 100
    support_size = 1

    X_init, y, beta, noise = multivariate_1D_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        sigma=0.1,
        rho=0.0,
        shuffle=False,
        seed=3,
    )

    # create a linear SVR estimator
    estimator = LinearSVR(C=1.0)

    # run the permutation test
    weight, weight_distribution = permutation_test(
        X_init, y, estimator=estimator, n_permutations=n_permutations
    )

    # test the shape of the output
    assert weight.shape == (n_features,)
    assert weight_distribution.shape == (n_permutations, n_features)
    # test weights
    assert not np.all(weight == 0.0)
    # test the distribution of the weights
    assert not np.all(weight_distribution == 0.0)
    assert not np.all(weight_distribution == weight)
    for i in range(n_permutations - 1):
        assert not np.all(weight_distribution[i, :] == weight_distribution[i + 1, :])

    # compute the p-values
    pval_corr, _ = permutation_test_pval(weight, weight_distribution)

    # expected p-values
    expected = 0.5 * np.ones(n_features)
    expected[:support_size] = 0.0

    # test
    assert_almost_equal(pval_corr, expected, decimal=1)


def test_permutation_test_with_fitted_estimator():
    """
    Testing if the weight of a fitted estimator is the same that the one obtained
    from the function _fit_and_weights.
    """

    # parameters for generating the data
    n_samples, n_features = 20, 50
    n_permutations = 100
    support_size = 1
    seed_permutation_test = 12

    X_init, y, beta, noise = multivariate_1D_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        sigma=0.1,
        rho=0.0,
        shuffle=False,
        seed=3,
    )

    # create a linear SVR estimator
    estimator = LinearSVR(C=1.0, random_state=0)
    estimator_fitted = clone(estimator).fit(X_init, y)

    # run the permutation test
    weight, weight_distribution = permutation_test(
        X_init,
        y,
        estimator=estimator,
        n_permutations=n_permutations,
        seed=seed_permutation_test,
    )

    # run the permutation test with the fitted estimator
    weight_fitted, weight_distribution_fitted = permutation_test(
        X_init,
        y,
        estimator=estimator_fitted,
        n_permutations=n_permutations,
        seed=seed_permutation_test,
    )

    # test the weights
    assert np.all(weight == weight_fitted)
    # test the distribution of the weights
    assert np.all(weight_distribution == weight_distribution_fitted)


def test_permutation_test_with_fitted_seed():
    """
    Testing if the weight of a fitted estimator is the same that the one obtained
    from the function _fit_and_weights.
    """

    # parameters for generating the data
    n_samples, n_features = 20, 50
    n_permutations = 100
    support_size = 1

    X_init, y, beta, noise = multivariate_1D_simulation(
        n_samples=n_samples,
        n_features=n_features,
        support_size=support_size,
        sigma=0.1,
        rho=0.0,
        shuffle=False,
        seed=3,
    )

    # create a linear SVR estimator
    estimator = LinearSVR(C=1.0, random_state=0)
    estimator_clone = clone(estimator)

    # run the permutation test
    weight, weight_distribution = permutation_test(
        X_init, y, estimator=estimator, n_permutations=n_permutations, seed=42
    )

    # run the permutation test with the fitted estimator
    weight_seed_diff, weight_distribution_seed_diff = permutation_test(
        X_init, y, estimator=estimator_clone, n_permutations=n_permutations, seed=10
    )

    # test the weights
    assert np.all(weight == weight_seed_diff)
    # test the distribution of the weights
    assert np.all(weight_distribution != weight_distribution_seed_diff)
