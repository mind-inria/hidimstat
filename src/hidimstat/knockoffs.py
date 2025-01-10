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

from .gaussian_knockoff import gaussian_knockoff_generation
from hidimstat.stat_tools import coef_diff_threshold


def preconfigure_estimator(estimator, X, X_tilde, y, n_lambdas=10):
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
    fdr=0.1,
    offset=1,
    estimator=LassoCV(n_jobs=None, verbose=0,max_iter=1000,
            cv=KFold(n_splits=5, shuffle=True, random_state=0),
            tol=1e-8),
    preconfigure_estimator = preconfigure_estimator,
    centered=True,
    cov_estimator=LedoitWolf(assume_centered=True),
    verbose=False,
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

    fdr : float, optional (default=0.1)
        The desired controlled False Discovery Rate (FDR) level.

    offset : int, 0 or 1, optional (default=1)
        The offset to calculate the knockoff threshold. An offset of 1 is equivalent to
        knockoff+.
            
    estimator : sklearn estimator instance, optional
        The estimator used for fitting the data and computing the test statistics.
        This can be any estimator with a `fit` method that accepts a 2D array and
        a 1D array, and a `coef_` attribute that returns a 1D array of coefficients.
        Examples include LassoCV, LogisticRegressionCV, and LinearRegression.

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
    n_samples, n_features = X.shape
    
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
    
    # Compute statistic base on a cross validation
    # original implementation:
    # https://github.com/msesia/knockoff-filter/blob/master/R/knockoff/R/stats_glmnet_cv.R
    X_ko = np.column_stack([X, X_tilde])
    if preconfigure_estimator is not None:
        preconfigure_estimator(estimator, X, X_tilde, y)
    estimator.fit(X_ko, y)
    if hasattr(estimator, 'coef_'):
        coef = np.ravel(estimator.coef_)
    elif hasattr(estimator, 'best_estimator_') and hasattr(estimator.best_estimator_, 'coef_'):
        coef = np.ravel(estimator.best_estimator_.coef_)  # for CV object
    test_score = np.abs(coef[:n_features]) - np.abs(coef[n_features:])
    
    # run the knockoff filter 
    threshold = coef_diff_threshold(test_score, fdr=fdr, offset=offset)
    selected = np.where(test_score >= threshold)[0]

    if verbose:
        return selected, threshold, test_score, X_tilde, estimator
    else:
        return selected
