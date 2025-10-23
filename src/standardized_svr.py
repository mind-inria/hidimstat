import numpy as np
from numpy.linalg import norm
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def standardized_svr(X, y, Cs=np.logspace(-7, 1, 9), n_jobs=1):
    """Cross-validated SVR

    Parameters
    -----------
    X : ndarray, shape (n_samples, n_features)
        Data.

    y : ndarray, shape (n_samples,)
        Target.

    Cs : ndarray, optional (default=np.logspace(-7, 1, 9))
        The linear SVR regularization parameter is set by cross-val running
        a grid search on the list of hyper-parameters contained in Cs.

    n_jobs : int or None, optional (default=1)
        Number of CPUs to use during the cross validation.

    Returns
    -------
    beta_hat : array, shape (n_features,)
        Estimated parameter vector.

    scale : ndarray, shape (n_features,)
        Value of the standard deviation of the parameters.
    """

    n_samples, n_features = X.shape

    steps = [("SVR", LinearSVR())]
    pipeline = Pipeline(steps)
    parameters = {"SVR__C": Cs}

    grid = GridSearchCV(pipeline, param_grid=parameters, n_jobs=n_jobs)
    grid.fit(X, y)

    beta_hat = grid.best_estimator_.named_steps["SVR"].coef_

    std = norm(beta_hat) / np.sqrt(n_features)
    scale = std * np.ones(beta_hat.size)

    return beta_hat, scale
