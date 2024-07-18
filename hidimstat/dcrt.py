"""Implementation of distillation Conditional Randomization Test, by Liu et
al. (2020) <https://arxiv.org/abs/2006.03980>. Currently only d0_CRT
is implemented.
"""

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
    fdr=0.1,
    estimated_coef=None,
    Sigma_X=None,
    cv=5,
    n_regus=20,
    max_iter=1000,
    use_cv=False,
    refit=False,
    loss="least_square",
    screening=True,
    screening_threshold=1e-1,
    scaled_statistics=False,
    statistic="residual",
    centered=True,
    alpha=None,
    solver="liblinear",
    fdr_control="bhq",
    n_jobs=1,
    verbose=False,
    joblib_verbose=0,
    ntree=100,
    type_prob="regression",
    random_state=2022,
):

    if centered:
        X = StandardScaler().fit_transform(X)

    _, n_features = X.shape

    if estimated_coef is None:
        if loss == "least_square":
            clf = LassoCV(
                cv=cv,
                n_jobs=n_jobs,
                n_alphas=n_regus * 2,
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

    # Screening step -- speed up computation of score function by only running
    # it later on estimated support set
    if screening:
        selection_set = np.setdiff1d(np.arange(n_features), non_selection)

        if selection_set.size == 0:
            if verbose:
                return np.array([]), np.ones(n_features), np.zeros(n_features)
            return np.array([])
    else:
        selection_set = np.arange(n_features)

    # refit with estimated support to possibly find better coeffcients
    # magnitude, as remarked in Ning & Liu 17
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
                    n_regus=5,
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
                    n_regus=n_regus,
                    ntree=ntree,
                    loss=loss,
                    type_prob=type_prob,
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
    elif statistic == "likelihood":
        pvals = stats.chi2.sf(ts, 1)

    threshold = fdr_threshold(pvals, fdr=fdr, method=fdr_control)
    selected = np.where(pvals <= threshold)[0]

    if verbose:
        return selected, pvals, ts

    return selected


def _x_distillation_lasso(
    X, idx, Sigma_X=None, cv=3, n_regus=100, alpha=None, use_cv=False, n_jobs=1
):

    n_samples = X.shape[0]
    X_minus_idx = np.delete(np.copy(X), idx, 1)

    if Sigma_X is None:
        if use_cv:
            clf = LassoCV(cv=cv, n_jobs=n_jobs, n_alphas=n_regus, random_state=0)
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
    n_regus=50,
    n_jobs=1,
    use_cv=False,
    alpha=None,
    fit_y=False,
):
    """Standard Lasso Distillation following Liu et al. (2020) section 2.4. Only
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
        n_regus=n_regus,
        n_jobs=n_jobs,
    )

    # Distill Y - calculate residual
    if use_cv:
        clf_null = LassoCV(cv=cv, n_jobs=n_jobs, n_alphas=n_regus, random_state=0)
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


def _optimal_reg_param(
    X,
    y,
    loss="least_square",
    n_regus=200,
    cv=5,
    n_jobs=1,
    solver="liblinear",
    max_iter=1000,
    tol=1e-5,
):
    """Which is a proportion of lambda_max. The idea is optimal_lambda is pretty
    close to each other across X_minus_j, so we just use cross-validation to
    find it once

    """
    X_minus_idx = np.delete(np.copy(X), 0, 1)

    if loss == "least_square":
        clf_null = LassoCV(
            cv=cv,
            n_jobs=n_jobs,
            n_alphas=n_regus,
            random_state=0,
            max_iter=max_iter,
            tol=tol,
        )
        clf_null.fit(X_minus_idx, y)
        optimal_lambda = clf_null.alpha_

    return optimal_lambda


def _rf_distillation(
    X,
    y,
    idx,
    Sigma_X=None,
    coef_full=None,
    cv=3,
    loss="least_square",
    n_regus=50,
    n_jobs=1,
    type_prob="regression",
    use_cv=False,
    ntree=100,
    alpha=None,
    random_state=2022,
):
    """
    Distillation using Random Forest
    """
    n_samples, _ = X.shape
    X_minus_idx = np.delete(np.copy(X), idx, 1)

    # Distill Y
    if type_prob == "regression":
        clf = RandomForestRegressor(n_estimators=ntree)
        clf.fit(X_minus_idx, y)
        eps_res = y - clf.predict(X_minus_idx)
        sigma2_y = np.mean(eps_res**2)

    elif type_prob == "classification":
        clf = RandomForestClassifier(n_estimators=ntree)
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
            n_regus=n_regus,
            n_jobs=n_jobs,
        )

        # T follows Gaussian distribution
        ts = np.dot(eps_res, X_res) / np.sqrt(n_samples * sigma2_X * sigma2_y)

    return ts
