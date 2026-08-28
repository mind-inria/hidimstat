import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from hidimstat import CFI
from hidimstat.statistical_tools.holdout_randomization_test import (
    holdout_randomization_test,
)

N_PERMUTATIONS = 200


def test_pvalue_over_permutations(rng):
    """Smoke test of the single-split setting: shapes and valid p-values."""
    n_features = 4
    loss_diff = rng.normal(loc=0.5, size=(n_features, N_PERMUTATIONS))

    result = holdout_randomization_test(loss_diff)

    assert result.statistic is None
    assert result.pvalue.shape == (n_features,)
    assert np.all(result.pvalue >= 1 / (N_PERMUTATIONS + 1))  # lower bound
    assert np.all(result.pvalue <= 1)

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


@pytest.mark.parametrize("statistical_test", ["hrt", "hrt-approx"])
@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [[300, 5, 2, 0.0, 1, 1.0, 10.0, 0.0]],
)
def test_holdout_randomization_test(data_generator, statistical_test):
    """
    Test that CFI combined with the holdout randomization test identifies the
    important features of a linear synthetic dataset. This pins down the
    direction of the test: a feature whose resampling increases the risk must
    get a *small* p-value.
    """
    X, y, important_features = data_generator
    n_features = X.shape[1]
    X_train, X_test = X[::2], X[1::2]
    y_train, y_test = y[::2], y[1::2]

    model = LinearRegression().fit(X_train, y_train)
    cfi = CFI(
        estimator=model,
        n_permutations=N_PERMUTATIONS,
        statistical_test=statistical_test,
        random_state=0,
    )
    cfi.fit(X_train, y_train)
    cfi.importance(X_test, y_test)

    alpha = 0.05
    assert cfi.pvalues_.shape == (n_features,)
    assert np.all(cfi.pvalues_ >= 1 / (N_PERMUTATIONS + 1))  # lower bound
    assert np.all(cfi.pvalues_[important_features] < alpha)
    assert np.all(cfi.pvalues_[~important_features] >= alpha)
