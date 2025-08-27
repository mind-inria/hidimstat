import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold


def preconfigure_LassoCV(estimator, X, X_tilde, y, n_alphas=20):
    """
    Configure the estimator for Model-X knockoffs.

    This function sets up the regularization path for the Lasso estimator
    based on the input data and the number of alphas to use. The regularization
    path is defined by a sequence of alpha values, which control the amount
    of shrinkage applied to the coefficient estimates.

    Parameters
    ----------
    estimator : sklearn.linear_model.LassoCV
        The Lasso estimator to configure.

    X : 2D ndarray (n_samples, n_features)
        The original design matrix.

    X_tilde : 2D ndarray (n_samples, n_features)
        The knockoff design matrix.

    y : 1D ndarray (n_samples, )
        The target vector.

    n_alphas : int, default=10
        The number of alpha values to use to instantiate the cross-validation.

    Returns
    -------
    estimator : sklearn.linear_model.LassoCV
        The configured estimator.

    Raises
    ------
    TypeError
        If estimator is not an instance of LassoCV.

    Notes
    -----
    The alpha values are calculated based on the combined design matrix [X, X_tilde].
    alpha_max is set to max(X_ko.T @ y)/(2*n_features).
    """
    if type(estimator).__name__ != "LassoCV":
        raise TypeError("You should not use this function to configure the estimator")

    n_features = X.shape[1]
    X_ko = np.column_stack([X, X_tilde])
    alpha_max = np.max(np.dot(X_ko.T, y)) / (2 * n_features)
    alphas = np.linspace(alpha_max * np.exp(-n_alphas), alpha_max, n_alphas)
    estimator.alphas = alphas
    return estimator


def lasso_statistic_with_sampling(
    X,
    X_tilde,
    y,
    lasso=LassoCV(
        n_jobs=1,
        verbose=0,
        max_iter=200000,
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        random_state=1,
        tol=1e-6,
    ),
    preconfigure_lasso=preconfigure_LassoCV,
):
    """
        Compute the Lasso Coefficient-Difference (LCD) statistic by comparing original and knockoff coefficients.

        This function fits a model on the concatenated original and knockoff features, then
        calculates test statistics based on the difference between coefficient magnitudes.
    Model-X Knockoff
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Original feature matrix.

        X_tilde : ndarray of shape (n_samples, n_features)
            Knockoff feature matrix.

        y : ndarray of shape (n_samples,)
            Target values.

        estimator : estimator object
            Scikit-learn estimator with fit() method and coef_ attribute.
            Common choices include LassoCV, LogisticRegressionCV.

        fdr : float
            Target false discovery rate level between 0 and 1.

        preconfigure_estimator : callable, default=None
            Optional function to configure estimator parameters before fitting.
            Called with arguments (estimator, X, X_tilde, y).

        Returns
        -------
        test_score : ndarray of shape (n_features,)
            Feature importance scores computed as |beta_j| - |beta_j'|
            where beta_j and beta_j' are original and knockoff coefficients.

        ko_thr : float
            Knockoff threshold value used for feature selection.

        selected : ndarray
            Indices of features with test_score >= ko_thr.

        Notes
        -----
        The test statistic follows Equation 1.7 in Barber & Candès (2015) and
        Equation 3.6 in Candès et al. (2018).
    """
    n_samples, n_features = X.shape
    X_ko = np.column_stack([X, X_tilde])
    if preconfigure_lasso is not None:
        lasso = preconfigure_lasso(lasso, X, X_tilde, y)
    lasso.fit(X_ko, y)
    if hasattr(lasso, "coef_"):
        coef = np.ravel(lasso.coef_)
    elif hasattr(lasso, "best_estimator_") and hasattr(lasso.best_estimator_, "coef_"):
        coef = np.ravel(lasso.best_estimator_.coef_)  # for CV object
    else:
        raise TypeError("estimator should be linear")
    # Equation 1.7 in barber2015controlling or 3.6 of candes2018panning
    test_score = np.abs(coef[:n_features]) - np.abs(coef[n_features:])

    return test_score


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
