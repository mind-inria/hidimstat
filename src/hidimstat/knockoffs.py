# -*- coding: utf-8 -*-
# Authors: Binh Nguyen <tuan-binh.nguyen@inria.fr>
"""
Implementation of Model-X knockoffs inference procedure, introduced in
Candes et. al. (2016) " Panning for Gold: Model-X Knockoffs for
High-dimensional Controlled Variable Selection"
<https://arxiv.org/abs/1610.02351>
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state
from joblib import Parallel, delayed

from hidimstat.gaussian_knockoff import gaussian_knockoff_generation, repeat_gaussian_knockoff_generation
from hidimstat.utils import fdr_threshold, quantile_aggregation


def preconfigure_estimator_LaccosCV(estimator, X, X_tilde, y, n_lambdas=10):
    """
    Configure the estimator for Model-X knockoffs.

    This function sets up the regularization path for the Lasso estimator
    based on the input data and the number of lambdas to use. The regularization
    path is defined by a sequence of lambda values, which control the amount
    of shrinkage applied to the coefficient estimates.

    Parameters:
    estimator: sklearn.linear_model._base.LinearModel
        An instance of a linear model estimator from sklearn.linear_model.
        This estimator will be used to fit the data and compute the test
        statistics. In this case, it should be an instance of LassoCV.

    X: 2D ndarray (n_samples, n_features)
        The original design matrix.

    X_tilde: 2D ndarray (n_samples, n_features)
        The knockoff design matrix.

    y: 1D ndarray (n_samples, )
        The target vector.

    n_lambdas: int, optional (default=10)
        The number of lambda values to use to instansiate the cross validation.

    Returns:
    None
        The function modifies the `estimator` object in-place.

    Raises:
    TypeError
        If the estimator is not an instance of LassoCV.

    Note:
    This function is specifically designed for the Model-X knockoffs procedure,
    which combines original and knockoff variables in the design matrix.
    """
    if type(estimator).__name__ != 'LassoCV':
        raise TypeError('You should not use this function for configure your estimator')

    n_features = X.shape[1]
    X_ko = np.column_stack([X, X_tilde])
    lambda_max = np.max(np.dot(X_ko.T, y)) / (2 * n_features)
    lambdas = np.linspace(lambda_max * np.exp(-n_lambdas), lambda_max, n_lambdas)
    estimator.alphas = lambdas


def model_x_knockoff(
    X,
    y,
    estimator=LassoCV(n_jobs=None, verbose=0,max_iter=1000,
            cv=KFold(n_splits=5, shuffle=True, random_state=0),
            tol=1e-8),
    preconfigure_estimator = preconfigure_estimator_LaccosCV,
    centered=True,
    cov_estimator=LedoitWolf(assume_centered=True),
    seed=None,
):
    """
    Model-X Knockoff

    This module implements the Model-X knockoff inference procedure, which is an approach
    to control the False Discovery Rate (FDR) based on Candes et al. (2017). The original
    implementation can be found at
    https://github.com/msesia/knockoff-filter/blob/master/R/knockoff/R/knockoff_filter.R

    Parameters
    ----------
    X : 2D ndarray (n_samples, n_features)
        The design matrix.

    y : 1D ndarray (n_samples, )
        The target vector.
  
    estimator : sklearn estimator instance, optional
        The estimator used for fitting the data and computing the test statistics.
        This can be any estimator with a `fit` method that accepts a 2D array and
        a 1D array, and a `coef_` attribute that returns a 1D array of coefficients.
        Examples include LassoCV, LogisticRegressionCV, and LinearRegression.
        Configuration example:
            LassoCV(alphas=lambdas, n_jobs=None, verbose=0, max_iter=1000,
                cv=KFold(n_splits=5, shuffle=True, random_state=0), tol=1e-8)
            LogisticRegressionCV(penalty="l1", max_iter=1000, solver="liblinear",
                cv=KFold(n_splits=5, shuffle=True, random_state=0), n_jobs=None, tol=1e-8)
            LogisticRegressionCV(penalty="l2", max_iter=1000, n_jobs=None,
                verbose=0, cv=KFold(n_splits=5, shuffle=True, random_state=0), tol=1e-8,)

    preconfigure_estimator : function, optional
        A function that configures the estimator for the Model-X knockoff procedure.
        If provided, this function will be called with the estimator, X, X_tilde, and y
        as arguments, and should modify the estimator in-place.

    centered : bool, optional (default=True)
        Whether to standardize the data before performing the inference procedure.

    cov_estimator : sklearn covariance estimator instance, optional
        The method used to estimate the empirical covariance matrix. This can be any
        estimator with a `fit` method that accepts a 2D array and a `covariance_`
        attribute that returns a 2D array of the estimated covariance matrix.
        Examples include LedoitWolf and GraphicalLassoCV.

    verbose : bool, optional (default=False)
        Whether to return additional information along with the selected variables.
        If True, the function will return a tuple containing the selected variables,
        the threshold, the test scores, and the knockoff design matrix. If False,
        the function will return only the selected variables.

    seed : int or None, optional (default=None)
        The random seed used to generate the Gaussian knockoff variables.
        Returns
        -------
        selected : 1D array, int
        A vector of indices of the selected variables.

    threshold : float
        The knockoff threshold.

    test_score : 1D array, (n_features, )
        A vector of test statistics.

    X_tilde : 2D array, (n_samples, n_features)
        The knockoff design matrix.

    References
    ----------
    .. footbibliography::
    """
    if centered:
        X = StandardScaler().fit_transform(X)

    # estimation of X distribution
    # original implementation:
    # https://github.com/msesia/knockoff-filter/blob/master/R/knockoff/R/create_second_order.R
    mu = X.mean(axis=0)
    sigma = cov_estimator.fit(X).covariance_

    # Create knockoff variables
    X_tilde = gaussian_knockoff_generation(
        X, mu, sigma, seed=seed
    )
    
    test_score = _stat_coefficient_diff(X, X_tilde, y, estimator, preconfigure_estimator)

    return test_score 


def model_x_knockoff_aggregation(
    X,
    y,
    estimator=LassoCV(n_jobs=None, verbose=0,max_iter=1000,
            cv=KFold(n_splits=5, shuffle=True, random_state=0),
            tol=1e-8),
    preconfigure_estimator = preconfigure_estimator_LaccosCV,
    centered=True,
    cov_estimator=LedoitWolf(assume_centered=True),
    joblib_verbose=0,
    n_bootstraps=25,
    n_jobs=1,
    random_state=None,
):
    assert n_bootstraps > 1, "the number of bootstraps should at least higher than 1"
    # unnecessary to have n_jobs > number of bootstraps
    n_jobs = min(n_bootstraps, n_jobs)
    parallel = Parallel(n_jobs, verbose=joblib_verbose)
    
    # get the seed for the different run
    if isinstance(random_state, (int, np.int32, np.int64)):
        rng = check_random_state(random_state)
    elif random_state is None:
        rng = check_random_state(0)
    else:
        raise TypeError("Wrong type for random_state")
    seed_list = rng.randint(1, np.iinfo(np.int32).max, n_bootstraps)
    
    if centered:
        X = StandardScaler().fit_transform(X)

    # estimation of X distribution
    mu = X.mean(axis=0)
    sigma = cov_estimator.fit(X).covariance_

    # Create knockoff variables
    X_tilde, (Mu_tilde, sigma_tilde_decompose)  = gaussian_knockoff_generation(
            X, mu, sigma, seed=seed_list[0], repeat=True
        )
    X_tildes = parallel(
        delayed(repeat_gaussian_knockoff_generation)(
            Mu_tilde, sigma_tilde_decompose, seed=seed
        )
        for seed in seed_list[1:]
    )
    X_tildes.insert(0, X_tilde)
    
    test_scores = parallel(
        delayed(_stat_coefficient_diff)(X, X_tildes[i], y, estimator, preconfigure_estimator)
        for i in range(n_bootstraps)
    )

    return test_scores
    

def model_x_knockoff_filter(test_score, fdr=0.1, offset=1, selection_only=True):
    """
    Calculate the p-values and return the selected variables based on the knockoff filter.

    Parameters
    ----------
    test_score : 1D array, (n_features, )
        A vector of test statistics.

    fdr : float, optional (default=0.1)
        The desired controlled False Discovery Rate (FDR) level.

    offset : int, 0 or 1, optional (default=1)
        The offset to calculate the knockoff threshold. An offset of 1 is equivalent to
        knockoff+.

    selection_only : bool, optional (default=True)
        Whether to return only the selected variables or additional information.
        If True, the function will return only the selected variables. If False,
        the function will return the selected variables, the threshold, and the test scores.

    Returns
    -------
    selected : 1D array, int
        A vector of indices of the selected variables.

    threshold : float
        The knockoff threshold.

    test_score : 1D array, (n_features, )
        A vector of test statistics.

    Notes
    -----
    This function calculates the knockoff threshold based on the test statistics and the
    desired FDR level. It then identifies the selected variables based on the threshold.
    """
    if offset not in (0, 1):
        raise ValueError("'offset' must be either 0 or 1")
    
    # run the knockoff filter
    threshold = _knockoff_threshold(test_score, fdr=fdr, offset=offset)
    selected = np.where(test_score >= threshold)[0]

    if selection_only:
        return selected
    else:
        return selected, threshold


def model_x_knockoff_pvalue(test_score, fdr=0.1, fdr_control="bhq", offset=1, selection_only=True):
    """
    This function implements the computation of the empirical p-values
    """
    pvals = _empirical_knockoff_pval(test_score, offset)
    threshold = fdr_threshold(pvals, fdr=fdr, method=fdr_control)
    selected = np.where(pvals <= threshold)[0]

    if selection_only:
        return selected
    else:
        return selected, pvals


def model_x_knockoff_bootstrap_e_value(test_scores, fdr=0.1, offset=1, selection_only=True):
        n_bootstraps = len(test_scores)
        evals = np.array(
            [_empirical_knockoff_eval(test_scores[i], fdr / 2, offset) for i in range(n_bootstraps)]
        )

        aggregated_eval = np.mean(evals, axis=0)
        threshold = fdr_threshold(aggregated_eval, fdr=fdr, method="ebh")
        selected = np.where(aggregated_eval >= threshold)[0]

        if selection_only:
            return selected
        else:
            return selected, aggregated_eval, evals


def model_x_knockoff_bootstrap_quantile(test_scores, fdr=0.1,  fdr_control="bhq", reshaping_function=None, adaptive_aggregation=False, gamma=0.5, gamma_min=0.05, offset=1, selection_only=True):
        n_bootstraps = len(test_scores)
        pvals = np.array(
            [_empirical_knockoff_pval(test_scores[i], offset) for i in range(n_bootstraps)]
        )

        aggregated_pval = quantile_aggregation(
            pvals, gamma=gamma, gamma_min=gamma_min, adaptive=adaptive_aggregation
        )

        threshold = fdr_threshold(
            aggregated_pval,
            fdr=fdr,
            method=fdr_control,
            reshaping_function=reshaping_function,
        )
        selected = np.where(aggregated_pval <= threshold)[0]

        if selection_only:
            return selected
        else:
            return selected, aggregated_pval, pvals


def _stat_coefficient_diff(X, X_tilde, y, estimator, preconfigure_estimator=None):
    # Compute statistic base on a cross validation
    # original implementation:
    # https://github.com/msesia/knockoff-filter/blob/master/R/knockoff/R/stats_glmnet_cv.R
    n_samples, n_features = X.shape
    X_ko = np.column_stack([X, X_tilde])
    if preconfigure_estimator is not None:
        preconfigure_estimator(estimator, X, X_tilde, y)
    estimator.fit(X_ko, y)
    if hasattr(estimator, 'coef_'):
        coef = np.ravel(estimator.coef_)
    elif hasattr(estimator, 'best_estimator_') and hasattr(estimator.best_estimator_, 'coef_'):
        coef = np.ravel(estimator.best_estimator_.coef_)  # for CV object
    test_score = np.abs(coef[:n_features]) - np.abs(coef[n_features:])
    return test_score


def _knockoff_threshold(test_score, fdr=0.1, offset=1):
    """
    Calculate the knockoff threshold based on the procedure stated in the
    article.
    
    original code:
    https://github.com/msesia/knockoff-filter/blob/master/R/knockoff/R/knockoff_filter.R

    Parameters
    ----------
    test_score : 1D ndarray, shape (n_features, )
        vector of test statistic

    fdr : float, optional
        desired controlled FDR(false discovery rate) level

    offset : int, 0 or 1, optional
        offset equals 1 is the knockoff+ procedure

    Returns
    -------
    threshold : float or np.inf
        threshold level
    """
    if offset not in (0, 1):
        raise ValueError("'offset' must be either 0 or 1")

    threshold_mesh = np.sort(np.abs(test_score[test_score != 0]))
    np.concatenate([[0], threshold_mesh, [np.inf]]) # if there is no solution, the threshold is inf
    # find the right value of t for getting a good fdr
    threshold = 0.
    for threshold in threshold_mesh:
        false_pos = np.sum(test_score <= -threshold)
        selected = np.sum(test_score >= threshold)
        if (offset + false_pos) / np.maximum(selected, 1) <= fdr:
            break
    return threshold

def _empirical_knockoff_pval(test_score, offset=1):
    """
    This function implements the computation of the empirical p-values
    from knockoff test
    """
    pvals = []
    n_features = test_score.size

    if offset not in (0, 1):
        raise ValueError("'offset' must be either 0 or 1")

    test_score_inv = -test_score
    for i in range(n_features):
        if test_score[i] <= 0:
            pvals.append(1)
        else:
            pvals.append(
                (offset + np.sum(test_score_inv >= test_score[i])) / n_features
            )

    return np.array(pvals)


def _empirical_knockoff_eval(test_score, fdr=0.1, offset=1):
    """
    This function implements the computation of the empirical e-values
    from knockoff test
    """
    evals = []
    n_features = test_score.size

    if offset not in (0, 1):
        raise ValueError("'offset' must be either 0 or 1")

    ko_thr = _knockoff_threshold(test_score, fdr=fdr, offset=offset)

    for i in range(n_features):
        if test_score[i] < ko_thr:
            evals.append(0)
        else:
            evals.append(n_features / (offset + np.sum(test_score <= -ko_thr)))

    return np.array(evals)