import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold

from hidimstat import CFI
from hidimstat.statistical_tools.holdout_randomization_test import (
    holdout_randomization_test,
)

N_PERMUTATIONS = 200


def _loss_diff(X_train, y_train, X_test, y_test):
    """
    Loss differences of one CFI fit, of shape (n_features, n_permutations):
    for each feature and each draw from the conditional distribution, the
    empirical risk on the sampled data minus the risk on the original data.
    """
    model = LinearRegression().fit(X_train, y_train)
    cfi = CFI(
        estimator=model,
        imputation_model_continuous=RidgeCV(),
        n_permutations=N_PERMUTATIONS,
        random_state=0,
    )
    cfi.fit(X_train, y_train)
    cfi.importance(X_test, y_test)
    return np.array(
        [
            cfi.loss_[j] - cfi.loss_reference_
            for j in range(cfi.n_features_groups_)
        ]
    )


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [[300, 5, 2, 0.0, 1, 1.0, 10.0, 0.0]],
)
def test_pvalue_over_permutations(data_generator):
    """Smoke test of the single-split setting: shapes and valid p-values."""
    X, y, _ = data_generator
    n_features = X.shape[1]
    loss_diff = _loss_diff(X[::2], y[::2], X[1::2], y[1::2])
    assert loss_diff.shape == (n_features, N_PERMUTATIONS)

    result = holdout_randomization_test(loss_diff)

    assert result.statistic is None
    assert result.pvalue.shape == (n_features,)
    assert np.all(result.pvalue >= 1 / (N_PERMUTATIONS + 1))  # lower bound
    assert np.all(result.pvalue <= 1)
    # the namedtuple is unpackable, like the other statistical tests
    statistic, pvalue = result
    assert statistic is None
    np.testing.assert_array_equal(pvalue, result.pvalue)

    # a single feature can be passed as a 1D array, as for the NB t-test
    single = holdout_randomization_test(loss_diff[0])
    assert single.pvalue.shape == (1,)
    assert single.pvalue[0] == result.pvalue[0]


def test_pvalue_over_folds(rng):
    """
    Test the two ways of combining the per-fold p-values: a Bonferroni
    correction over the folds (the default), or a sum of the loss differences
    over the folds before a single p-value is computed (``approx=True``).
    """
    n_features, n_folds = 4, 3
    loss_diff = rng.normal(loc=0.5, size=(n_features, N_PERMUTATIONS, n_folds))

    corrected = holdout_randomization_test(loss_diff).pvalue
    approx = holdout_randomization_test(loss_diff, approx=True).pvalue

    for pvalue in (corrected, approx):
        assert pvalue.shape == (n_features,)
        assert np.all(pvalue >= 1 / (N_PERMUTATIONS + 1))  # lower bound
        assert np.all(pvalue <= 1)  # the Bonferroni correction is clipped

    # algorithm 4 pools the folds instead of correcting for them, which is
    # less conservative when the signal is consistent across the folds
    assert np.all(approx < corrected)


def test_wrong_dimension():
    """Test that only 1D, 2D and 3D inputs are accepted."""
    with pytest.raises(ValueError, match="must be 1D, 2D, or 3D"):
        holdout_randomization_test(np.zeros((2, 3, 4, 5)))


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [[300, 5, 2, 0.0, 1, 1.0, 10.0, 0.0]],
)
def test_holdout_randomization_test(data_generator):
    """
    Test that the holdout randomization test identifies the important features
    of a linear synthetic dataset, in the single-split setting and in the
    cross-validated one with both combinations of the per-fold p-values.
    """
    X, y, important_features = data_generator
    n_features = X.shape[1]
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    loss_diff_list = [
        _loss_diff(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        for train_idx, test_idx in cv.split(X)
    ]
    # shape (n_features, n_permutations, n_folds)
    loss_diff_array = np.stack(loss_diff_list, axis=-1)
    assert loss_diff_array.shape == (
        n_features,
        N_PERMUTATIONS,
        cv.get_n_splits(),
    )

    alpha = 0.05
    single_split = holdout_randomization_test(loss_diff_list[0]).pvalue
    corrected = holdout_randomization_test(loss_diff_array).pvalue
    approx = holdout_randomization_test(loss_diff_array, approx=True).pvalue

    for pvalue in (single_split, corrected, approx):
        assert pvalue.shape == (n_features,)
        assert np.all(pvalue[important_features] < alpha)
        assert np.all(pvalue[~important_features] >= alpha)
