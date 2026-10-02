import numpy as np
from sklearn.utils.validation import check_is_fitted

from hidimstat._utils.baselines import (
    _LOCIBaselineClassifier,
    _LOCIBaselineRegressor,
)


def test_baseline_regressor_fitted(rng):
    """
    Verify that fit of baseline regressor works, and verify the output predictions.
    """
    X = rng.random((100, 5))
    y = rng.random(100)

    baseline_regressor = _LOCIBaselineRegressor()
    baseline_regressor.fit(X, y)

    check_is_fitted(baseline_regressor)
    np.testing.assert_equal(baseline_regressor.baseline_mean_, np.mean(y))

    X = rng.random((20, 5))
    y_pred = baseline_regressor.predict(X)
    assert y_pred.shape == (X.shape[0],)
    np.testing.assert_equal(
        y_pred, np.full((X.shape[0],), np.mean(y), dtype=float)
    )


def test_baseline_classifier_fitted(rng):
    """
    Verify that fit of baseline classifier works, and verify the output predictions.
    """
    X = rng.random((100, 5))
    y = rng.random(100)

    baseline_classifier = _LOCIBaselineClassifier()
    baseline_classifier.fit(X, y)

    check_is_fitted(baseline_classifier)
    y_values_, y_counts_ = np.unique(y, return_counts=True)
    np.testing.assert_equal(
        baseline_classifier.baseline_mean_, y_counts_[1] / y.shape[0]
    )

    X = rng.random((20, 5))
    y_pred = baseline_classifier.predict_proba(X)
    assert y_pred.shape == (X.shape[0], len(y_values_))
    expected_output = np.full(
        (X.shape[0], len(y_values_)), y_counts_[1] / y.shape[0], dtype=float
    )
    np.testing.assert_equal(y_pred, expected_output)
    np.testing.assert_equal(
        baseline_classifier.decision_function(X), expected_output
    )
