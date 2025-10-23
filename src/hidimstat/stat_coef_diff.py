# -*- coding: utf-8 -*-
# Authors: Binh Nguyen <tuan-binh.nguyen@inria.fr>

import numpy as np
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import KFold

# from sklearn.linear_model._coordinate_descent import _alpha_grid
# from sklearn.model_selection import GridSearchCV


def stat_coef_diff(
    X,
    X_tilde,
    y,
    method="lasso_cv",
    n_splits=5,
    n_jobs=1,
    n_lambdas=10,
    n_iter=1000,
    joblib_verbose=0,
    return_coef=False,
    solver="liblinear",
    seed=0,
):
    """Calculate test statistic by doing estimation with Cross-validation on
    concatenated design matrix [X X_tilde] to find coefficients [beta
    beta_tilda]. The test statistic is then:

                        W_j =  abs(beta_j) - abs(beta_tilda_j)

    with j = 1, ..., n_features

    Parameters
    ----------
    X : 2D ndarray (n_samples, n_features)
        Original design matrix

    X_tilde : 2D ndarray (n_samples, n_features)
        Knockoff design matrix

    y : 1D ndarray (n_samples, )
        Response vector

    loss : str, optional
        if the response vector is continuous, the loss used should be
        'least_square', otherwise
        if the response vector is binary, it should be 'logistic'

    n_splits : int, optional
        number of cross-validation folds

    solver : str, optional
        solver used by sklearn function LogisticRegressionCV

    n_regu : int, optional
        number of regulation used in the regression problem

    return_coef : bool, optional
        return regression coefficient if set to True

    Returns
    -------
    test_score : 1D ndarray (n_features, )
        vector of test statistic

    coef: 1D ndarray (n_features * 2, )
        coefficients of the estimation problem
    """

    n_features = X.shape[1]
    X_ko = np.column_stack([X, X_tilde])
    lambda_max = np.max(np.dot(X_ko.T, y)) / (2 * n_features)
    lambdas = np.linspace(lambda_max * np.exp(-n_lambdas), lambda_max, n_lambdas)

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    estimator = {
        "lasso_cv": LassoCV(
            alphas=lambdas,
            n_jobs=n_jobs,
            verbose=joblib_verbose,
            max_iter=n_iter,
            cv=cv,
        ),
        "logistic_l1": LogisticRegressionCV(
            penalty="l1", max_iter=n_iter, solver=solver, cv=cv, n_jobs=n_jobs, tol=1e-8
        ),
        "logistic_l2": LogisticRegressionCV(
            penalty="l2",
            max_iter=n_iter,
            n_jobs=n_jobs,
            verbose=joblib_verbose,
            cv=cv,
            tol=1e-8,
        ),
    }

    try:
        clf = estimator[method]
    except KeyError:
        print(f"{method} is not a valid estimator")

    clf.fit(X_ko, y)

    try:
        coef = np.ravel(clf.coef_)
    except AttributeError:
        coef = np.ravel(clf.best_estimator_.coef_)  # for GridSearchCV object

    test_score = np.abs(coef[:n_features]) - np.abs(coef[n_features:])

    if return_coef:
        return test_score, coef

    return test_score


def _coef_diff_threshold(test_score, fdr=0.1, offset=1):
    """Calculate the knockoff threshold based on the procedure stated in the
    article.

    Parameters
    ----------
    test_score : 1D ndarray, shape (n_features, )
        vector of test statistic

    fdr : float, optional
        desired controlled FDR level

    offset : int, 0 or 1, optional
        offset equals 1 is the knockoff+ procedure

    Returns
    -------
    thres : float or np.inf
        threshold level
    """
    if offset not in (0, 1):
        raise ValueError("'offset' must be either 0 or 1")

    t_mesh = np.sort(np.abs(test_score[test_score != 0]))
    for t in t_mesh:
        false_pos = np.sum(test_score <= -t)
        selected = np.sum(test_score >= t)
        if (offset + false_pos) / np.maximum(selected, 1) <= fdr:
            return t

    return np.inf
