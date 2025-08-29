import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold


def lasso_statistic(
    X,
    y,
    lasso=LassoCV(
        n_jobs=1,
        verbose=0,
        max_iter=200000,
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        tol=1e-6,
    ),
    n_alphas=0,
):
    """
    Compute Lasso statistic using feature coefficients.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data matrix.
    y : array-like of shape (n_samples,)
        The target values.
    lasso : estimator, default=LassoCV(n_jobs=None, verbose=0, max_iter=200000, cv=KFold(n_splits=5, shuffle=True, random_state=0), tol=1e-6)
        The Lasso estimator to use for computing the test statistic.
    n_alphas : int, default=0
        Number of alpha values to test for Lasso regularization path.
        If 0, uses the default alpha sequence from the estimator.
    Returns
    -------
    coef : ndarray
        Lasso coefficients for each feature.
    Raises
    ------
    TypeError
        If the provided estimator does not have coef_ attribute or is not linear.
    """
    if n_alphas != 0:
        alpha_max = np.max(np.dot(X.T, y)) / (X.shape[1])
        alphas = np.linspace(alpha_max * np.exp(-n_alphas), alpha_max, n_alphas)
        lasso.alphas = alphas
    lasso.fit(X, y)
    if hasattr(lasso, "coef_"):
        coef = np.ravel(lasso.coef_)
    elif hasattr(lasso, "best_estimator_") and hasattr(lasso.best_estimator_, "coef_"):
        coef = np.ravel(lasso.best_estimator_.coef_)  # for CV object
    else:
        raise TypeError("estimator should be linear")
    return coef
