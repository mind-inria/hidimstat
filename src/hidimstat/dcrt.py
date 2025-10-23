import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler

from hidimstat.utils import _lambda_max, fdr_threshold


def dcrt_zero(
    X,
    y,
    fdr=0.1,
    estimated_coef=None,
    Sigma_X=None,
    use_cv=False,
    cv=5,
    n_alphas=20,
    alpha=None,
    max_iter=1000,
    refit=False,
    loss="least_square",
    screening=True,
    screening_threshold=1e-1,
    scaled_statistics=False,
    statistic="residual",
    centered=True,
    fdr_control="bhq",
    n_jobs=1,
    verbose=False,
    joblib_verbose=0,
    ntree=100,
    problem_type="regression",
    random_state=2022,
):
    """
    This function implements the Conditional Randomization Test of
    :footcite:t:`candesPanningGoldModelX2017` accelerated with the distillation
    process `dcrt_zero` in the work by :footcite:t:`liuFastPowerfulConditional2021`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target values (class labels in classification, real numbers in
        regression).
    fdr : float, default=0.1
        The desired controlled FDR level.
    estimated_coef : array-like of shape (n_features,)
        The array of the corresponding coefficients for the features.
    Sigma_X : {array-like, sparse matrix} of shape (n_features, n_features)
        The covariance matrix of X.
    use_cv : bool, default=False
        Whether to apply cross-validation for the distillation with Lasso.
    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy. Possible inputs for
        cv are:
        - None, to use the default 5-fold cross-validation,
        - int, to specify the number of folds.
        - CV splitter,
        - An iterable yielding (train, test) splits as arrays of indices.
    n_alphas : int, default=20
        The number of alphas along the regularization path for
        sklearn.linear_model.LassoCV().
    alpha : float, default=None
        Constant that multiplies the L1 term, controlling regularization
        strength. alpha must be a non-negative float i.e. in [0, inf).
    max_iter : int, default=1000
        The maximum number of iterations.
    refit : bool, default=False
        If estimated_coef is not provided, whether to refit with the estimated support set to possibly find better
        coefficients magnitude.
    loss : str, default="least_square"
        The loss function used for the distillation of X.
    screening : bool, default=True
        Speed up the computation of score function by only running it later on
        estimated support set.
    screening_threshold : float, default=1e-1
        The threshold for the estimated support set. screening_threshold must be
        a non-negative float in (0, 100).
    scaled_statistics : bool, default=False
        Whether to scale the test statistics.
    statistic : str, default='residual'
        The estimator used to distill the outcome based on the remaining
        variables after removing the variable of interest. The options include:
        - "residual" for the Lasso learner
        - "randomforest" for the Random Forest learner
    centered : bool, default=True
        Whether to standard scale the input features using
        sklearn.preprocessing.StandardScaler().
    fdr_control : srt, default="bhq"
        The control method for False Discovery Rate (FDR). The options include:
        - "bhq" for Standard Benjamini-Hochberg procedure
        - "bhy" for Benjamini-Hochberg-Yekutieli procedure
        - "ebh" for e-BH procedure
    n_jobs : int, default=1
        The number of workers for parallel processing.
    verbose : bool, default=False
        Whether to return the corresponding p-values and test statistics of the
        variables along with the list of selected variables.
    joblib_versobe : int, default=0
       The verbosity level of joblib: if non zero, progress messages are
       printed. Above 50, the output is sent to stdout. The frequency of the
       messages increases with the verbosity level. If it more than 10, all
       iterations are reported.
    ntree : int, default=100
        The number of trees for the distillation using the Random Forest
        learner.
    problem_type : str, default='regression'
        A classification or a regression problem.
    random_state : int, default=2023
        Fixing the seeds of the random generator.

    Returns
    -------
    selected : 1D array, int
        The vector of index of selected variables.
    pvals: 1D array, float
        The vector of the corresponding p-values.
    ts: 1D array, float
        The vector of the corresponding test statistics.

    References
    ----------
    .. footbibliography::
    """
    if centered:
        X = StandardScaler().fit_transform(X)

    _, n_features = X.shape

    if estimated_coef is None:
        if loss == "least_square":
            clf = LassoCV(
                cv=cv,
                n_jobs=n_jobs,
                n_alphas=n_alphas * 2,
                tol=1e-6,
                fit_intercept=False,
                random_state=0,
                max_iter=max_iter,
            )
        else:
            raise ValueError(f"{loss} loss is not supported.")
        clf.fit(X, y)
        coef_X_full = np.ravel(clf.coef_)
    else:
        coef_X_full = estimated_coef
        screening_threshold = 100

    # noisy estimated coefficients is set to 0.0

    non_selection = np.where(
        np.abs(coef_X_full)
        <= np.percentile(np.abs(coef_X_full), 100 - screening_threshold)
    )[0]
    coef_X_full[non_selection] = 0.0

    if screening:
        selection_set = np.setdiff1d(np.arange(n_features), non_selection)

        if selection_set.size == 0:
            if verbose:
                return np.array([]), np.ones(n_features), np.zeros(n_features)
            return np.array([])
    else:
        selection_set = np.arange(n_features)

    if refit and estimated_coef is None and selection_set.size < n_features:
        clf_refit = clone(clf)
        clf_refit.fit(X[:, selection_set], y)
        coef_X_full[selection_set] = np.ravel(clf_refit.coef_)

    # Distillation & calculate score function
    if statistic == "residual":
        # For distillation of X it should always be least_square loss
        if loss == "least_square":
            results = Parallel(n_jobs, verbose=joblib_verbose)(
                delayed(_lasso_distillation_residual)(
                    X,
                    y,
                    idx,
                    coef_X_full,
                    Sigma_X=Sigma_X,
                    cv=cv,
                    use_cv=use_cv,
                    alpha=alpha,
                    n_jobs=1,
                    n_alphas=5,
                )
                for idx in selection_set
            )
        else:
            raise ValueError(f"{loss} loss is not supported.")
    elif statistic == "randomforest":
        if loss == "least_square":
            results = Parallel(n_jobs, verbose=joblib_verbose)(
                delayed(_rf_distillation)(
                    X,
                    y,
                    idx,
                    Sigma_X=Sigma_X,
                    cv=3,
                    use_cv=use_cv,
                    alpha=alpha,
                    n_jobs=1,
                    n_alphas=n_alphas,
                    ntree=ntree,
                    loss=loss,
                    problem_type=problem_type,
                    random_state=random_state,
                )
                for idx in selection_set
            )
        else:
            raise ValueError(f"{loss} loss is not supported.")
    else:
        raise ValueError(f"{statistic} statistic is not supported.")
    ts = np.zeros(n_features)
    ts[selection_set] = np.array([i for i in results])

    if scaled_statistics:
        ts = (ts - np.mean(ts)) / np.std(ts)

    if statistic in ["residual", "randomforest"]:
        pvals = np.minimum(2 * stats.norm.sf(np.abs(ts)), 1)

    threshold = fdr_threshold(pvals, fdr=fdr, method=fdr_control)
    selected = np.where(pvals <= threshold)[0]

    if verbose:
        return selected, pvals, ts

    return selected


def _x_distillation_lasso(
    X, idx, Sigma_X=None, cv=3, n_alphas=100, alpha=None, use_cv=False, n_jobs=1
):
    """
    This function applies the distillation of the variable of interest with the
    remaining variables using the lasso
    """
    n_samples = X.shape[0]
    X_minus_idx = np.delete(np.copy(X), idx, 1)

    if Sigma_X is None:
        if use_cv:
            clf = LassoCV(cv=cv, n_jobs=n_jobs, n_alphas=n_alphas, random_state=0)
            clf.fit(X_minus_idx, X[:, idx])
            alpha = clf.alpha_
        else:
            if alpha is None:
                alpha = 0.1 * _lambda_max(
                    X_minus_idx, X[:, idx], use_noise_estimate=False
                )
            clf = Lasso(alpha=alpha, fit_intercept=False)
            clf.fit(X_minus_idx, X[:, idx])

        X_res = X[:, idx] - clf.predict(X_minus_idx)
        sigma2_X = np.linalg.norm(X_res) ** 2 / n_samples + alpha * np.linalg.norm(
            clf.coef_, ord=1
        )

    else:
        Sigma_temp = np.delete(np.copy(Sigma_X), idx, 0)
        b = Sigma_temp[:, idx]
        A = np.delete(np.copy(Sigma_temp), idx, 1)
        coefs_X = np.linalg.solve(A, b)
        X_res = X[:, idx] - np.dot(X_minus_idx, coefs_X)
        sigma2_X = Sigma_X[idx, idx] - np.dot(
            np.delete(np.copy(Sigma_X[idx, :]), idx), coefs_X
        )

    return X_res, sigma2_X


def _lasso_distillation_residual(
    X,
    y,
    idx,
    coef_full,
    Sigma_X=None,
    cv=3,
    n_alphas=50,
    alpha=None,
    n_jobs=1,
    use_cv=False,
    fit_y=False,
):
    """
    Standard Lasso Distillation following Liu et al. (2020) section 2.4. Only
    works for least square loss regression.
    """
    n_samples, _ = X.shape

    X_minus_idx = np.delete(np.copy(X), idx, 1)

    # Distill X
    X_res, sigma2_X = _x_distillation_lasso(
        X,
        idx,
        Sigma_X,
        cv=cv,
        use_cv=use_cv,
        alpha=alpha,
        n_alphas=n_alphas,
        n_jobs=n_jobs,
    )

    # Distill Y - calculate residual
    if use_cv:
        clf_null = LassoCV(cv=cv, n_jobs=n_jobs, n_alphas=n_alphas, random_state=0)
    else:
        if alpha is None:
            alpha = 0.5 * _lambda_max(X_minus_idx, y, use_noise_estimate=False)
        clf_null = Lasso(alpha=alpha, fit_intercept=False)

    if fit_y:
        clf_null.fit(X_minus_idx, y)
        coef_minus_idx = clf_null.coef_
    else:
        coef_minus_idx = np.delete(np.copy(coef_full), idx)

    eps_res = y - X_minus_idx.dot(coef_minus_idx)
    sigma2_y = np.mean(eps_res**2)

    # T follows Gaussian distribution
    ts = np.dot(eps_res, X_res) / np.sqrt(n_samples * sigma2_X * sigma2_y)

    return ts


def _rf_distillation(
    X,
    y,
    idx,
    Sigma_X=None,
    cv=3,
    loss="least_square",
    n_alphas=50,
    alpha=None,
    n_jobs=1,
    problem_type="regression",
    use_cv=False,
    ntree=100,
    random_state=None,
):
    """
    This function implements the distillation process using Random Forest
    """
    n_samples, _ = X.shape
    X_minus_idx = np.delete(np.copy(X), idx, 1)

    # Distill Y
    if problem_type == "regression":
        clf = RandomForestRegressor(n_estimators=ntree, random_state=random_state)
        clf.fit(X_minus_idx, y)
        eps_res = y - clf.predict(X_minus_idx)
        sigma2_y = np.mean(eps_res**2)

    elif problem_type == "classification":
        clf = RandomForestClassifier(n_estimators=ntree, random_state=random_state)
        clf.fit(X_minus_idx, y)
        eps_res = y - clf.predict_proba(X_minus_idx)[:, 1]
        sigma2_y = np.mean(eps_res**2)

    # Distill X
    if loss == "least_square":
        X_res, sigma2_X = _x_distillation_lasso(
            X,
            idx,
            Sigma_X,
            cv=cv,
            use_cv=use_cv,
            alpha=alpha,
            n_alphas=n_alphas,
            n_jobs=n_jobs,
        )

        # T follows Gaussian distribution
        ts = np.dot(eps_res, X_res) / np.sqrt(n_samples * sigma2_X * sigma2_y)

    return ts
