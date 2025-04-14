import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler

from hidimstat._utils.utils import _alpha_max
from hidimstat.statistical_tools.multiple_testing import fdr_threshold


def dcrt_zero(
    X,
    y,
    estimated_coef=None,
    sigma_X=None,
    params_lasso_screening={
        "alpha": None,
        "n_alphas": 10,
        "alphas": None,
        "alpha_max_fraction": 0.5,
        "cv": 5,
        "tol": 1e-6,
        "max_iter": 1000,
        "fit_intercept": False,
        "selection": "cyclic",
    },
    params_lasso_distillation_x=None,
    params_lasso_distillation_y=None,
    refit=False,
    screening=True,
    screening_threshold=1e-1,
    statistic="residual",
    centered=True,
    n_jobs=1,
    joblib_verbose=0,
    fit_y=False,
    n_tree=100,
    problem_type="regression",
    random_state=2022,
):
    """
    Implements distilled conditional randomization test (dCRT) without interactions.

    A faster version of the Conditional Randomization Test
    :footcite:t:`candes2018panning` using the distillation process from
    :footcite:t:`liu2022fast`. Based on original implementation at:
    https://github.com/moleibobliu/Distillation-CRT/

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data
    y : array-like of shape (n_samples,)
        Target values
    estimated_coef : array-like of shape (n_features,), optional
        Pre-computed feature coefficients
    sigma_X : array-like of shape (n_features, n_features), optional
        Covariance matrix of X
    params_lasso_screening : dict
        Parameters for main Lasso estimation or crossvalidation Lasso, including:
        - alpha : float, optional - L1 regularization strength. If None, determined by CV.
        - n_alphas : int, default=0 - Number of alphas for cross-validation.
        - alphas : array-like, default=None - List of alpha values to try in CV.
        - alpha_max_fraction : float, default=0.5 - Scale factor for alpha_max.
        For other parameters see :py:func:LassoCV, here is some advise configuration
        - cv : int, default=5 - Number of cross-validation folds.
        - tol : float, default=1e-6 - Tolerance for optimization.
        - max_iter : int, default=1000 - Maximum iterations.
        - fit_intercept : bool, default=False - Whether to fit intercept.
        - selection : str, default='cyclic' - Feature selection method.
    params_lasso_distillation_x : dict, optional
        Parameters for X distillation Lasso. Defaults to params_lasso_screening.
    params_lasso_distillation_y : dict, optional
        Parameters for y distillation Lasso. Defaults to params_lasso_screening.
    refit : bool, default=False
        Whether to refit on estimated support set
    screening : bool, default=True
        Whether to screen variables
    screening_threshold : float, default=0.1
        Threshold for variable screening (0-100)
    statistic : {'residual', 'random_forest'}, default='residual'
        Learning method for outcome distillation
    centered : bool, default=True
        Whether to standardize features
    n_jobs : int, default=1
        Number of parallel jobs
    joblib_verbose : int, default=0
        Verbosity level
    fit_y : bool, default=False
        Whether to fit y using selected features
    n_tree : int, default=100
        Number of trees for random forest
    problem_type : {'regression', 'classification'}, default='regression'
        Type of learning problem
    random_state : int, default=2022
        Random seed

    Returns
    -------
    selection_features : ndarray of shape (n_features,)
        Boolean mask of selected features
    X_res : ndarray of shape (n_selected, n_samples)
        Residuals after X distillation
    sigma2 : ndarray of shape (n_selected,)
        Estimated residual variances
    y_res : ndarray of shape (n_selected, n_samples)
        Response residuals

    References
    ----------
    .. footbibliography::
    """
    if centered:
        X_ = StandardScaler().fit_transform(X)
    else:
        X_ = X
    y_ = y  # avoid modifying the original y

    _, n_features = X_.shape

    ## Screening of variables for accelarate dCRT
    if estimated_coef is None:
        # base on the Theorem 2 of :footcite:t:`liu2022fast`, the rule of screening
        # is based on a cross-validated lasso
        clf_screening, alpha_screening = _fit_lasso(
            X_,
            y_,
            n_jobs=n_jobs,
            random_state=random_state,
            **params_lasso_screening,
        )
        # update the alpha value from the estiamtor on all values
        params_lasso_screening["alpha"] = alpha_screening
        if params_lasso_distillation_x is not None:
            params_lasso_distillation_x["alpha"] = alpha_screening
        if params_lasso_distillation_y is not None:
            params_lasso_distillation_y["alpha"] = alpha_screening
        coef_X_full = np.ravel(clf_screening.coef_)
    else:
        coef_X_full = estimated_coef
        screening_threshold = 100  # remove the screening process

    # noisy estimated coefficients is set to 0.0
    non_selection = np.where(
        np.abs(coef_X_full)
        <= np.percentile(np.abs(coef_X_full), 100 - screening_threshold)
    )[0]
    # optimisation to reduce the number of elements different to zeros
    coef_X_full[non_selection] = 0.0

    # select the variables for the screening
    if screening:
        selection_set = np.setdiff1d(np.arange(n_features), non_selection)

        if selection_set.size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
    else:
        non_selection = []
        selection_set = np.arange(n_features)

    # Refit the model with the estimated support set
    if refit and estimated_coef is None and selection_set.size < n_features:
        clf_refit = clone(clf_screening)
        clf_refit.fit(X_[:, selection_set], y_)
        coef_X_full[selection_set] = np.ravel(clf_refit.coef_)

    ## Distillation & calculate
    if statistic == "residual":
        # For distillation of X use least_square loss
        results = Parallel(n_jobs, verbose=joblib_verbose)(
            delayed(_lasso_distillation_residual)(
                X_,
                y_,
                idx,
                coef_full=coef_X_full,
                sigma_X=sigma_X,
                params_lasso_distillation_x=(
                    params_lasso_distillation_x
                    if params_lasso_distillation_x is not None
                    else params_lasso_screening
                ),
                params_lasso_distillation_y=(
                    params_lasso_distillation_y
                    if params_lasso_distillation_y is not None
                    else params_lasso_screening
                ),
                fit_y=fit_y,
                n_jobs=1,  # the function is already called in parallel
                random_state=random_state,
            )
            for idx in selection_set
        )
    elif statistic == "random_forest":
        # For distillation of X use least_square loss
        results = Parallel(n_jobs, verbose=joblib_verbose)(
            delayed(_rf_distillation)(
                X_,
                y_,
                idx,
                sigma_X=sigma_X,
                n_tree=n_tree,
                problem_type=problem_type,
                n_jobs=1,  # the function is already called in parallel
                random_state=random_state,
                params_lasso_distillation_x=(
                    params_lasso_distillation_x
                    if params_lasso_distillation_y is not None
                    else params_lasso_screening
                ),
            )
            for idx in selection_set
        )
    else:
        raise ValueError(f"{statistic} statistic is not supported.")

    # contatenate result
    selection_features = np.ones((n_features,), dtype=bool)
    selection_features[non_selection] = 0
    X_residual = np.array([result[0] for result in results])
    sigma2 = np.array([result[1] for result in results])
    y_residual = np.array([result[2] for result in results])
    return selection_features, X_residual, sigma2, y_residual


def dcrt_pvalue(
    selection_features,
    X_res,
    sigma2,
    y_res,
    fdr=0.1,
    fdr_control="bhq",
    reshaping_function=None,
    scaled_statistics=False,
):
    """
    Calculate p-values and identify significant features using the dCRT test statistics.

    This function processes the results from dCRT to identify statistically significant
    features while controlling for false discoveries. It assumes test statistics follow
    a Gaussian distribution.

    Parameters
    ----------
    selection_features : ndarray of shape (n_features,)
        Boolean mask indicating which features were selected for testing
    X_res : ndarray of shape (n_selected, n_samples)
        Residuals from feature distillation
    sigma2 : ndarray of shape (n_selected,)
        Estimated residual variances for each tested feature
    y_res : ndarray of shape (n_selected, n_samples)
        Response residuals for each tested feature
    fdr : float, default=0.1
        Target false discovery rate level (0 < fdr < 1)
    fdr_control : {'bhq', 'bhy', 'ebh'}, default='bhq'
        Method for FDR control:
        - 'bhq': Benjamini-Hochberg procedure
        - 'bhy': Benjamini-Hochberg-Yekutieli procedure
        - 'ebh': e-BH procedure
    reshaping_function : callable, optional
        Reshaping function for the 'bhy' method
    scaled_statistics : bool, default=False
        Whether to standardize test statistics before computing p-values

    Returns
    -------
    selected_variables : ndarray
        Indices of features deemed significant
    pvals : ndarray of shape (n_features,)
        P-values for all features (including unselected ones)
    ts : ndarray of shape (n_features,)
        test statistics following a standard normal distribution for all features

    Notes
    -----
    The function computes test statistics as correlations between residuals,
    optionally scales them, and converts to p-values using a Gaussian null.
    Multiple testing correction is applied to control FDR at the specified level.
    """
    n_features = selection_features.shape[0]
    n_samples = X_res.shape[1]

    ts_selected_variables = [
        np.dot(y_res[i], X_res[i])
        / np.sqrt(n_samples * sigma2[i] * np.mean(y_res[i] ** 2))
        for i in range(X_res.shape[0])
    ]

    if scaled_statistics:
        ts_selected_variables = (
            ts_selected_variables - np.mean(ts_selected_variables)
        ) / np.std(ts_selected_variables)

    # get the results
    ts = np.zeros(n_features)
    ts[selection_features] = ts_selected_variables

    # for residual and random_forest, the test statistics follows Gaussian distribution
    pvals = np.minimum(2 * stats.norm.sf(np.abs(ts)), 1)

    threshold = fdr_threshold(
        pvals, fdr=fdr, method=fdr_control, reshaping_function=reshaping_function
    )
    selected_variables = np.where(pvals <= threshold)[0]

    return selected_variables, pvals, ts


def _x_distillation_lasso(
    X,
    idx,
    sigma_X=None,
    n_jobs=1,
    random_state=0,
    params_lasso_distillation_x=None,
):
    """
    Distill variable X[:, idx] using Lasso regression on remaining variables.

    This function implements the distillation process to estimate the conditional
    distribution of X[:, idx] given the remaining variables, using either Lasso
    regression or a known covariance matrix.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples.
    idx : int
        Index of the variable to be distilled.
    sigma_X : {array-like, sparse matrix} of shape (n_features, n_features), default=None
        The covariance matrix of X. If provided, used instead of Lasso regression.
    n_jobs : int, default=1
        Number of CPUs to use for cross-validation.
    random_state : int, default=0
        Random seed for reproducibility.
    params_lasso_distillation_x : dict
        Parameters for main Lasso estimation or crossvalidation Lasso, including:
        - alpha : float, optional - L1 regularization strength. If None, determined by CV.
        - n_alphas : int, default=0 - Number of alphas for cross-validation.
        - alphas : array-like, default=None - List of alpha values to try in CV.
        - alpha_max_fraction : float, default=0.5 - Scale factor for alpha_max.

    Returns
    -------
    X_res : ndarray of shape (n_samples,)
        The residuals after distillation.
    sigma2 : float
        The estimated variance of the residuals.
    """
    n_samples = X.shape[0]
    X_minus_idx = np.delete(np.copy(X), idx, 1)

    if sigma_X is None:
        # Distill X with least square loss
        # configure Lasso and determine the alpha
        clf, alpha = _fit_lasso(
            X_minus_idx,
            X[:, idx],
            n_jobs=n_jobs,
            random_state=random_state,
            **params_lasso_distillation_x,
        )

        # get the residuals
        X_res = X[:, idx] - clf.predict(X_minus_idx)
        # compute the variance of the residuals
        # In the original paper and implementation, the term:
        #  alpha * np.linalg.norm(clf.coef_, ord=1)
        # is not present and has been added without any reference actually
        sigma2 = np.linalg.norm(X_res) ** 2 / n_samples + alpha * np.linalg.norm(
            clf.coef_, ord=1
        )

    else:
        # Distill X with sigma_X
        sigma_temp = np.delete(np.copy(sigma_X), idx, 0)
        b = sigma_temp[:, idx]
        A = np.delete(np.copy(sigma_temp), idx, 1)
        coefs_X = np.linalg.solve(A, b)
        X_res = X[:, idx] - np.dot(X_minus_idx, coefs_X)
        sigma2 = sigma_X[idx, idx] - np.dot(
            np.delete(np.copy(sigma_X[idx, :]), idx), coefs_X
        )

    return X_res, sigma2


def _lasso_distillation_residual(
    X,
    y,
    idx,
    coef_full=None,
    sigma_X=None,
    n_jobs=1,
    fit_y=False,
    random_state=42,
    params_lasso_distillation_x={
        "alpha": None,
        "n_alphas": 10,
        "alphas": None,
        "alpha_max_fraction": 0.5,
    },
    params_lasso_distillation_y={
        "alpha": None,
        "n_alphas": 10,
        "alphas": None,
        "alpha_max_fraction": 0.5,
    },
):
    """
    Standard Lasso Distillation for least squares regression.

    This function implements the distillation process following :footcite:t:`liu2022fast`
    section 2.3. It distills both X[:, idx] and y to compute test statistics.
    It's based on least square loss regression.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target values.
    idx : int
        Index of the variable to be tested.
    coef_full : array-like of shape (n_features,), default=None
        Pre-computed coefficients for y prediction. If None, computed via Lasso.
    sigma_X : {array-like, sparse matrix} of shape (n_features, n_features), default=None
        The covariance matrix of X.
    n_jobs : int, default=1
        Number of CPUs to use.
    fit_y : bool, default=False
        Whether to fit y using Lasso when coef_full is None.
    random_state : int, default=42
        Random seed for reproducibility.
    params_lasso_distillation_x : dict
        Parameters for main Lasso estimation or crossvalidation Lasso, including:
        - alpha : float, optional - L1 regularization strength. If None, determined by CV.
        - n_alphas : int, default=0 - Number of alphas for cross-validation.
        - alphas : array-like, default=None - List of alpha values to try in CV.
        - alpha_max_fraction : float, default=0.5 - Scale factor for alpha_max.
    params_lasso_distillation_y : dict
        Same as params_lasso_distillation_x.

    Returns
    -------
    X_res : ndarray of shape (n_samples,)
        The residuals after X distillation.
    sigma2 : float
        The estimated variance of the residuals.
    y_res : ndarray of shape (n_samples,)
        The residuals after y distillation.

    References
    ----------
    .. footbibliography::
    """
    X_minus_idx = np.delete(np.copy(X), idx, 1)

    # Distill X with least square loss
    X_res, sigma2 = _x_distillation_lasso(
        X,
        idx,
        sigma_X,
        n_jobs=n_jobs,
        random_state=random_state + 2,  # avoid the same seed as the main function
        params_lasso_distillation_x=params_lasso_distillation_x,
    )

    # Distill Y - calculate residual
    # get the coefficients
    if fit_y:
        clf_null, alpha_null = _fit_lasso(
            X_minus_idx,
            y,
            n_jobs=n_jobs,
            random_state=random_state,
            **params_lasso_distillation_y,
        )
        coef_minus_idx = clf_null.coef_
    elif coef_full is not None:
        coef_minus_idx = np.delete(np.copy(coef_full), idx)
    else:
        raise ValueError("Either fit_y is true or coeff_full must be provided.")

    # compute the residuals
    y_res = y - X_minus_idx.dot(coef_minus_idx)

    return X_res, sigma2, y_res


def _rf_distillation(
    X,
    y,
    idx,
    sigma_X=None,
    n_jobs=1,
    problem_type="regression",
    n_tree=100,
    random_state=42,
    params_lasso_distillation_x={
        "alpha": None,
        "n_alphas": 10,
        "alphas": None,
        "alpha_max_fraction": 0.5,
    },
):
    """
    Random Forest based distillation for both regression and classification.

    This function implements the distillation process using Random Forest for y
    and Lasso for X[:, idx]. It supports both regression and binary classification
    problems.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target values (class labels or regression targets).
    idx : int
        Index of the variable to be tested.
    sigma_X : {array-like, sparse matrix} of shape (n_features, n_features), default=None
        The covariance matrix of X.
    n_jobs : int, default=1
        Number of CPUs to use.
    problem_type : {'regression', 'classification'}, default='regression'
        The type of prediction problem.
    n_tree : int, default=100
        Number of trees in the Random Forest.
    random_state : int, default=42
        Random seed for reproducibility.
    params_lasso_distillation_x : dict
        Parameters for main Lasso estimation or crossvalidation Lasso, including:
        - alpha : float, optional - L1 regularization strength. If None, determined by CV.
        - n_alphas : int, default=0 - Number of alphas for cross-validation.
        - alphas : array-like, default=None - List of alpha values to try in CV.
        - alpha_max_fraction : float, default=0.5 - Scale factor for alpha_max.

    Returns
    -------
    X_res : ndarray of shape (n_samples,)
        The residuals after X distillation.
    sigma2 : float
        The estimated variance of the residuals.
    y_res : ndarray of shape (n_samples,)
        The residuals after y distillation.

    Notes
    -----
    For classification, the function uses probability predictions from
    RandomForestClassifier and assumes binary classification (uses class 1
    probability only).
    """
    n_samples, _ = X.shape
    X_minus_idx = np.delete(np.copy(X), idx, 1)

    # Distill X with least square loss
    X_res, sigma2 = _x_distillation_lasso(
        X,
        idx,
        sigma_X,
        n_jobs=n_jobs,
        random_state=random_state + 2,  # avoid the same seed as the main function
        params_lasso_distillation_x=params_lasso_distillation_x,
    )

    # Distill Y
    # get the residuals
    if problem_type == "regression":
        clf = RandomForestRegressor(
            n_estimators=n_tree, random_state=random_state, n_jobs=n_jobs
        )
        clf.fit(X_minus_idx, y)
        y_res = y - clf.predict(X_minus_idx)

    elif problem_type == "classification":
        clf = RandomForestClassifier(
            n_estimators=n_tree, random_state=random_state, n_jobs=n_jobs
        )
        clf.fit(X_minus_idx, y)
        y_res = y - clf.predict_proba(X_minus_idx)[:, 1]  # IIABDFI

    return (
        X_res,
        sigma2,
        y_res,
    )


def _fit_lasso(
    X,
    y,
    n_jobs,
    alpha,
    alphas,
    n_alphas,
    alpha_max_fraction,
    random_state,
    **params,
):
    """
    Fits a LASSO regression model with optional cross-validation for alpha selection.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data
    y : array-like of shape (n_samples,)
        Target values
    n_jobs : int
        Number of CPUs for cross-validation
    alpha : float
        Constant that multiplies the L1 term. If None and alphas/n_alphas not provided,
        alpha is set to alpha_max_fraction * alpha_max.
    alphas : array-like
        List of alphas where to compute the models. If None and n_alphas > 0,
        alphas are set automatically.
    n_alphas : int
        Number of alphas along the regularization path. Ignored if alphas is provided.
    alpha_max_fraction : float
        Fraction of alpha_max to use when alpha, alphas, and n_alphas are not provided.
    random_state : int, RandomState instance or None
        Random seed for reproducibility
    **params : dict
        Additional parameters for Lasso/LassoCV

    Returns
    -------
    clf : estimator
        Fitted Lasso/LassoCV model
    alpha : float
        Selected alpha value

    Notes
    -----
    Uses cross-validation to select the best alpha if alphas or n_alphas provided.
    Otherwise, uses a single alpha value either provided or computed from alpha_max_fraction.
    """
    if alphas is not None or n_alphas > 0:
        clf = LassoCV(
            n_alphas=n_alphas,
            alphas=alphas,
            n_jobs=n_jobs,
            random_state=random_state,
            **params,
        )
        clf.fit(X, y)
        alpha = clf.alpha_
    else:
        if alpha is None:
            alpha = alpha_max_fraction * _alpha_max(X, y)
        clf = Lasso(
            alpha=alpha,
            random_state=random_state,
            **params,
        )
        clf.fit(X, y)
    return clf, alpha
