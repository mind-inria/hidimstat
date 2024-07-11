"""
Test the importance functions module
"""

import numpy as np
import pandas as pd
from hidimstat.BBI import BlockBasedImportance
from sklearn.datasets import make_classification, make_regression

# Fixing the random seed
rng = np.random.RandomState(2024)


def _generate_data(n_samples=100, n_features=10, problem_type="regression", seed=2024):

    if problem_type == "regression":
        X, y = make_regression(
            n_samples=n_samples,
            noise=0.2,
            n_features=n_features,
            random_state=seed,
        )
    else:
        X, y = make_classification(
            n_samples=n_samples,
            n_classes=2,
            n_informative=5,
            n_features=n_features,
            random_state=seed,
        )
        y = np.array([str(i) for i in y])

    X = pd.DataFrame(X, columns=[f"col{i+1}" for i in range(n_features)])
    # Nominal variables
    X["Val1"] = rng.choice(["Car", "Moto", "Velo", "Plane", "Boat"], size=n_samples)
    X["Val2"] = rng.choice(["Car_1", "Moto_1", "Velo_1", "Plane_1"], size=n_samples)
    # Ordinal variables
    X["Val3"] = rng.choice(np.arange(1, 7), size=n_samples)
    X["Val4"] = rng.choice(np.arange(10), size=n_samples)

    variables_categories = {"nominal": ["Val1", "Val2"], "ordinal": ["Val3", "Val4"]}
    return X, y, variables_categories


def test_compute_loco():
    pass
