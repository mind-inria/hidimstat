import warnings
import numpy as np
from joblib import Parallel, delayed
from hidimstat.utils import _lambda_max, fdr_threshold
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler


def dcrt_zero(
    X,
    y,
    estimated_coef=None,
    sigma_X=None,
    use_cv=False,
    cv=5,
    n_alphas=20,
    alpha=None,
    tol=1e-6,
    max_iter=1000,
    refit=False,
    screening=True,
    screening_threshold=1e-1,
    statistic="residual",
    centered=True,
    n_jobs=1,
    joblib_verbose=0,
    fit_y=False,
    ntree=100,
    problem_type="regression",
    random_state=2022,
):
    """
    Implements distilled conditional randomization test (dCRT) without interactions.

    A faster version of the Conditional Randomization Test :cite:`candes2018panning` using the distillation
    process from :cite:`liu2022fast`. Based on original implementation at:
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
    use_cv : bool, default=False
        Whether to use cross-validation for Lasso
    cv : int, default=5
        Number of cross-validation folds
    tol : float, default=1e-6
        Solver tolerance for the cross valisation of Lasso
    n_alphas : int, default=20
        Number of alphas for Lasso path
    alpha : float, optional
        L1 regularization strength
    max_iter : int, default=1000
        Maximum iterations
    refit : bool, default=False
        Whether to refit on estimated support set
    screening : bool, default=True
        Whether to screen variables
    screening_threshold : float, default=0.1
        Threshold for variable screening (0-100)
    statistic : {'residual', 'randomforest'}, default='residual'
        Learning method for outcome distillation
    centered : bool, default=True
        Whether to standardize features
    n_jobs : int, default=1
        Number of parallel jobs
    joblib_verbose : int, default=0
        The verbosity level of joblib: if non zero, progress messages are
        printed. Above 50, the output is sent to stdout. The frequency of the
        messages increases with the verbosity level. If it more than 10, all
        iterations are reported.
    fit_y : bool, default=False
        Whether to fit y using selected features
    ntree : int, default=100
        Number of trees for random forest
    problem_type : {'regression', 'classification'}, default='regression'
        Type of learning problem
    random_state : int, default=2022
        Random seed

    Returns
    -------
    selection_features : ndarray
        Boolean mask of selected features
    X_res : ndarray
        Residuals after distillation
    sigma2_X : ndarray
        Estimated residual variances
    y_res : ndarray
        Target residuals after distillation

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
        # base on the Theorem 2 of :cite:`liu2022fast`, the rule of screening
        # is based on a cross-validated of lasso 
        clf = LassoCV(
            cv=cv,
            n_jobs=n_jobs,
            n_alphas=n_alphas * 2,  # TODO: Why * 2 ?
            tol=tol,
            fit_intercept=False,
            random_state=random_state + 1,  # avoid the same seed as the main function
            max_iter=max_iter,
        )
        clf.fit(X_, y_)
        coef_X_full = np.ravel(clf.coef_)
    else:
        coef_X_full = estimated_coef
        screening_threshold = 100  # remove the screening process

    # noisy estimated coefficients is set to 0.0
    non_selection = np.where(
        np.abs(coef_X_full)
        <= np.percentile(np.abs(coef_X_full), 100 - screening_threshold)
    )[0]
    coef_X_full[non_selection] = 0.0 #TODO this should be after the screening process ????

    # select the variables for the screening
    if screening:
        selection_set = np.setdiff1d(np.arange(n_features), non_selection)

        if selection_set.size == 0:
            return np.array([])
    else:
        non_selection = []
        selection_set = np.arange(n_features)

    # Refit the model with the estimated support set
    if refit and estimated_coef is None and selection_set.size < n_features:
        clf_refit = clone(clf)
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
                cv=cv,
                use_cv=use_cv,
                alpha=alpha,
                n_jobs=1,  # the function is already called in parallel
                n_alphas=n_alphas,
                fit_y=fit_y,
                random_state=random_state,
            )
            for idx in selection_set
        )
    elif statistic == "randomforest":
        # For distillation of X use least_square loss
        results = Parallel(n_jobs, verbose=joblib_verbose)(
            delayed(_rf_distillation)(
                X_,
                y_,
                idx,
                sigma_X=sigma_X,
                cv=cv,
                use_cv=use_cv,
                alpha=alpha,
                n_jobs=1,  # the function is already called in parallel
                n_alphas=n_alphas,
                ntree=ntree,
                problem_type=problem_type,
                random_state=random_state,
            )
            for idx in selection_set
        )
    else:
        raise ValueError(f"{statistic} statistic is not supported.")

    # contatenate result
    selection_features = np.ones((n_features,), dtype=bool)
    selection_features[non_selection] = 0
    X_res = np.array([i[0] for i in results]) 
    sigma2_X = np.array([i[1] for i in results]) 
    y_res = np.array([i[2] for i in results])
    
    return selection_features, X_res, sigma2_X, y_res


def dcrt_pvalue(
    selection_features, X_res, sigma2_X, y_res,  fdr=0.1, fdr_control="bhq", selection_only=True, reshaping_function=None, 
    scaled_statistics=False,
):
    """
    Computes p-values for feature importance using a Gaussian distribution following the recommendation of :cite:`liu2022fast`

    Parameters
    ----------
    selection_features : array-like of shape (n_features,)
        Boolean mask indicating which features were selected for testing
    X_res : array-like
        Residuals of X after distillation 
    sigma2_X : array-like
        Estimated residual variances
    y_res : array-like  
        Residuals of y after distillation
    fdr : float, default=0.1
        Target false discovery rate level
    fdr_control : {'bhq', 'bhy', 'ebh'}, default='bhq'
        Method for FDR control:
        - 'bhq': Benjamini-Hochberg procedure
        - 'bhy': Benjamini-Hochberg-Yekutieli procedure  
        - 'ebh': e-BH procedure
    selection_only : bool, default=True
        If True, return only selected variables. If False, also return p-values and statistics
    reshaping_function : callable, default=None
        Function applied in Benjamini-Hochberg-Yekutieli procedure
    scaled_statistics : bool, default=False 
        Whether to standardize test statistics

    Returns
    -------
    variables_important : ndarray
        Indices of selected variables
    pvals : ndarray, optional
        P-values if selection_only=False  
    ts : ndarray, optional
        Test statistics if selectionn_only=False
    
    References
    ----------
    .. footbibliography::
    """
    n_features = selection_features.shape[0]
    n_samples = X_res.shape[1]
    
    # compute the test statistic for selected features
    # this based on the equation for the p-value of the page 284 of `liu2022fast`
    ts_selected_variables = [np.dot(y_res[i, :], X_res[i, :]) / np.sqrt(n_samples * sigma2_X[i] * np.mean(y_res[i, :]**2)) for i in range(X_res.shape[0])]

    if scaled_statistics:
        ts_selected_variables = (ts_selected_variables - np.mean(ts_selected_variables)) / np.std(ts_selected_variables)

    # define the test statistic for all features
    ts = np.zeros(n_features)
    ts[selection_features] = ts_selected_variables

    # for residual and randomforest, the test statistics follows Gaussian distribution
    # this based on the equation for the p-value of the page 284 of `liu2022fast`
    pvals = np.minimum(2 * stats.norm.sf(np.abs(ts)), 1)

    threshold = fdr_threshold(
        pvals, fdr=fdr, method=fdr_control, reshaping_function=reshaping_function
    )
    variables_important = np.where(pvals <= threshold)[0]

    if selection_only:
        return variables_important
    else:
        return variables_important, pvals, ts


def _x_distillation_lasso(
    X,
    idx,
    sigma_X=None,
    cv=3,
    n_alphas=100,
    alpha=None,
    use_cv=False,
    n_jobs=1,
    random_state=0,
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
    cv : int, cross-validation generator or iterable, default=3
        Determines the cross-validation splitting strategy for LassoCV.
    n_alphas : int, default=100
        Number of alphas along the regularization path for LassoCV.
    alpha : float, default=None
        The regularization strength for Lasso. If None, determined automatically.
    use_cv : bool, default=False
        Whether to use cross-validation to select alpha.
    n_jobs : int, default=1
        Number of CPUs to use for cross-validation.
    random_state : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    X_res : ndarray of shape (n_samples,)
        The residuals after distillation.
    sigma2_X : float
        The estimated variance of the residuals.
    """
    n_samples = X.shape[0]
    X_minus_idx = np.delete(np.copy(X), idx, 1)

    if sigma_X is None:
        # Distill X with least square loss
        # configure Lasso and determine the alpha
        if use_cv:
            clf = LassoCV(
                cv=cv, n_alphas=n_alphas, n_jobs=n_jobs, random_state=random_state
            )
            clf.fit(X_minus_idx, X[:, idx])
            alpha = clf.alpha_
        else:
            if alpha is None:
                alpha = 0.1 * _lambda_max(
                    X_minus_idx, X[:, idx], use_noise_estimate=False
                )  # TODO: why 0.1 ?
            clf = Lasso(
                alpha=alpha,
                fit_intercept=False,
                random_state=random_state,
            )
            clf.fit(X_minus_idx, X[:, idx])

        # get the residuals
        X_res = X[:, idx] - clf.predict(X_minus_idx)
        # compute the variance of the residuals
        sigma2_X = np.linalg.norm(X_res) ** 2 / n_samples + alpha * np.linalg.norm(
            clf.coef_, ord=1
        )
        #TODO the calculation in original implementation doesn't not include alpha

    else:
        # Distill X with sigma_X
        sigma_temp = np.delete(np.copy(sigma_X), idx, 0)
        b = sigma_temp[:, idx]
        A = np.delete(np.copy(sigma_temp), idx, 1)
        # compute the coefficient of the linear estimator
        coefs_X = np.linalg.solve(A, b)
        # compute the residual and the variance
        X_res = X[:, idx] - np.dot(X_minus_idx, coefs_X)
        sigma2_X = sigma_X[idx, idx] - np.dot(
            np.delete(np.copy(sigma_X[idx, :]), idx), coefs_X
        )

    return X_res, sigma2_X


def _lasso_distillation_residual(
    X,
    y,
    idx,
    coef_full=None,
    sigma_X=None,
    cv=3,
    n_alphas=50,
    alpha=None,
    n_jobs=1,
    use_cv=False,
    fit_y=False,
    alpha_max_fraction=0.5,
    random_state=42,
):
    """
    Standard Lasso Distillation for least squares regression.

    This function implements the distillation process following :cite:`liu2022fast`
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
    cv : int, default=3
        Number of folds for cross-validation.
    n_alphas : int, default=50
        Number of alphas along the regularization path.
    alpha : float, default=None
        The regularization strength. If None, determined automatically.
    n_jobs : int, default=1
        Number of CPUs to use.
    use_cv : bool, default=False
        Whether to use cross-validation for selecting alpha.
    fit_y : bool, default=False
        Whether to fit y using Lasso when coef_full is None.
    alpha_max_fraction : float, default=0.5
        Fraction of lambda_max to use when determining alpha.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_res : ndarray
        Residuals of X after distillation
    sigma2_X : float
        Estimated residual variance
    y_res : ndarray 
        Residuals of y after distillation

    References
    ----------
    .. footbibliography::
    """
    X_minus_idx = np.delete(np.copy(X), idx, 1)

    # Distill X with least square loss
    X_res, sigma2_X = _x_distillation_lasso(
        X,
        idx,
        sigma_X,
        cv=cv,
        use_cv=use_cv,
        alpha=alpha,
        n_alphas=n_alphas,
        n_jobs=n_jobs,
        random_state=random_state + 2,  # avoid the same seed as the main function
    )

    # Distill Y - calculate residual
    # get the coefficients
    if fit_y:
        # configure Lasso
        if use_cv:
            clf_null = LassoCV(
                cv=cv,
                n_alphas=n_alphas,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        else:
            if alpha is None:
                alpha = alpha_max_fraction * _lambda_max(
                    X_minus_idx, y, use_noise_estimate=False
                )
            clf_null = Lasso(
                alpha=alpha,
                fit_intercept=False,
                n_jobs=n_jobs,
                random_state=random_state,
            )

        clf_null.fit(X_minus_idx, y)
        coef_minus_idx = clf_null.coef_
    elif coef_full is not None:
        coef_minus_idx = np.delete(np.copy(coef_full), idx)
    else:
        raise ValueError("Either fit_y is true or coeff_full must be provided.")

    # compute the residuals
    y_res = y - X_minus_idx.dot(coef_minus_idx)

    return X_res, sigma2_X, y_res

def _rf_distillation(
    X,
    y,
    idx,
    sigma_X=None,
    cv=3,
    n_alphas=50,
    alpha=None,
    n_jobs=1,
    problem_type="regression",
    use_cv=False,
    ntree=100,
    random_state=42,
):
    """
    Random Forest based distillation for both regression and classification.

    This function implements the distillation process from :cite:`liu2022fast` using Random Forest for y
    and Lasso for X[:, idx]. It supports both regression and binary classification problems.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target values (class labels or regression targets).
    idx : int
        Index of the variable to be tested.
    sigma_X : {array-like, sparse matrix} of shape (n_features, n_features), default=None
        The covariance matrix of X. If provided, used instead of Lasso regression.
    cv : int, default=3
        Number of folds for cross-validation in X distillation.
    n_alphas : int, default=50
        Number of alphas for Lasso path in X distillation.
    alpha : float, default=None
        Regularization strength for X distillation. If None, determined automatically.
    n_jobs : int, default=1
        Number of CPUs to use.
    problem_type : {'regression', 'classification'}, default='regression'
        The type of prediction problem.
    use_cv : bool, default=False
        Whether to use cross-validation for X distillation.
    ntree : int, default=100
        Number of trees in the Random Forest.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_res : ndarray
        Residuals of X after distillation
    sigma2_X : float  
        Estimated residual variance
    y_res : ndarray
        Residuals of y after distillation. For classification, uses probability 
        difference from RandomForestClassifier prediction.

    References
    ----------
    .. footbibliography::
    """
    X_minus_idx = np.delete(np.copy(X), idx, 1)

    # Distill X with least square loss
    X_res, sigma2_X = _x_distillation_lasso(
        X,
        idx,
        sigma_X,
        cv=cv,
        use_cv=use_cv,
        alpha=alpha,
        n_alphas=n_alphas,
        n_jobs=n_jobs,
        random_state=random_state + 2,  # avoid the same seed as the main function
    )

    # Distill Y
    # get the residuals
    if problem_type == "regression":
        clf = RandomForestRegressor(
            n_estimators=ntree, random_state=random_state, n_jobs=n_jobs
        )
        clf.fit(X_minus_idx, y)
        y_res = y - clf.predict(X_minus_idx)

    elif problem_type == "classification":
        clf = RandomForestClassifier(
            n_estimators=ntree, random_state=random_state, n_jobs=n_jobs
        )
        clf.fit(X_minus_idx, y)
        y_res = (
            y - clf.predict_proba(X_minus_idx)[:, 1]
        )  #IIABDFI
        warnings.warn(UserWarning('Binary classification residuals are computed as (y - P(y=1|X)), assuming y in {0,1} and P(y=1|X) is probability prediction.'))

    return X_res, sigma2_X, y_res,
