from sklearn.datasets import make_classification, make_regression
import numpy as np
import pandas as pd


def generate_data(
    n_samples=200,
    n_features=10,
    problem_type="regression",
    seed=2024,
):
    """
    This function generates the synthetic data used in the different tests.
    
    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate, by default 200
    n_features : int, optional
        Number of features to generate, by default 10
    problem_type : str, optional
        Type of problem to generate, by default "regression" (options: "regression", "classification")
    seed : int, optional
        Random seed, by default 2024

    Returns
    -------
    X : pd.DataFrame
        Data matrix
    y : np.array
        Target vector
    grps : np.array
        Group vector
    """
    rng = np.random.default_rng(seed)
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
        #y = np.array([str(i) for i in y])

    X = pd.DataFrame(X, columns=[f"col{i+1}" for i in range(n_features)])

    return X, y