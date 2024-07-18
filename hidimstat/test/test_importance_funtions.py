"""
Test the importance functions module
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from hidimstat.importance_functions import compute_loco

# Fixing the random seed
rng = np.random.RandomState(2024)


def _generate_data(n_samples=100, n_features=2, problem_type="regression", seed=2024):
    """
    This function generates the synthetic data used in the different tests.
    """
    if problem_type == "regression":
        X, y = make_regression(
            n_samples=n_samples,
            noise=0.2,
            n_features=n_features,
            random_state=seed,
        )
    else:
        n_features = max(n_features, 5)
        X, y = make_classification(
            n_samples=n_samples,
            n_classes=2,
            n_features=n_features,
            random_state=seed,
        )
        y = np.array([str(i) for i in y])

    X = pd.DataFrame(X, columns=[f"col{i+1}" for i in range(n_features)])
    return X, y


def test_compute_loco_reg():
    """
    This function tests the application of the Leave-One-Covariate-Out (LOCO)
    method with a Multi-Layer Perceptron (MLP) or Random Forest (RF) learner
    under a regression case
    """
    X, y = _generate_data(problem_type="regression")

    # DNN
    results_reg_dnn = compute_loco(X, y, problem_type="regression", use_dnn=True)
    assert len(results_reg_dnn) == 2
    assert len(results_reg_dnn["importance_score"]) == X.shape[1]
    assert len(results_reg_dnn["p_value"]) == X.shape[1]

    # RF
    results_reg_rf = compute_loco(X, y, problem_type="regression", use_dnn=False)
    assert len(results_reg_rf) == 2
    assert len(results_reg_rf["importance_score"]) == X.shape[1]
    assert len(results_reg_rf["p_value"]) == X.shape[1]


def test_compute_loco_class():
    """
    This function tests the application of the Leave-One-Covariate-Out (LOCO)
    method with a Multi-Layer Perceptron (MLP) or Random Forest (RF) learner
    under a classification case
    """
    X, y = _generate_data(problem_type="classification")

    # DNN
    results_class_dnn = compute_loco(X, y, problem_type="classification", use_dnn=True)
    assert len(results_class_dnn) == 2
    assert len(results_class_dnn["importance_score"]) == X.shape[1]
    assert len(results_class_dnn["p_value"]) == X.shape[1]

    # RF
    results_class_rf = compute_loco(X, y, problem_type="classification", use_dnn=False)
    assert len(results_class_rf) == 2
    assert len(results_class_rf["importance_score"]) == X.shape[1]
    assert len(results_class_rf["p_value"]) == X.shape[1]
