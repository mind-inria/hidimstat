import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve, toeplitz
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
from sklearn.model_selection import KFold


def reid(X, y, eps=1e-2, tol=1e-4, max_iter=10000, n_split=5, n_jobs=1, seed=0):
    """
    FIXME: citation
    Smoothly Clipped Absolute Deviation Penalty (SCAD) of Fan and Li (2001)
    
    FIXME: citation
    Estimation of noise standard deviation using the most promissing procedure
    of Reid et al. (2016) [1] by comparison. 

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data.

    y : ndarray, shape (n_samples,)
        Target.

    eps: float, optional (default=1e-2)
        Length of the cross-validation path.
        eps=1e-2 means that alpha_min / alpha_max = 1e-2.

    tol : float, optional (default=1e-4)
        The tolerance for the optimization: if the updates are smaller
        than `tol`, the optimization code checks the dual gap for optimality
        and continues until it is smaller than `tol`.

    max_iter : int, optional (default=1e4)
        The maximum number of iterations.
    
    n_split : int, optional (default=5)
        Number of splits in the KFold object used to cross-validate LassoCV.

    n_jobs : int or None, optional (default=1)
        Number of CPUs to use during the cross validation.

    seed: int, optional (default=0)
        Seed passed in the KFold object which is used to cross-validate
        LassoCV. This seed controls the partitioning randomness.

    Returns
    -------
    sigma_hat : float
        Estimated noise standard deviation.

    beta_hat : array, shape (n_features,)
        Estimated parameter vector.

    References
    ----------
    .. [1] Reid, S., Tibshirani, R., & Friedman, J. (2016). A study of error
           variance estimation in lasso regression. Statistica Sinica, 35-67.
       [2] Fan, J., Guo, S., & Hao, N. (2012). Variance estimation using refitted 
           cross-validation in ultrahigh dimensional regression. Journal of the Royal 
           Statistical Society Series B: Statistical Methodology, 74(1), 37-65.
           
    """

    X_ = np.asarray(X)
    n_samples, n_features = X_.shape

    # check if max_iter is large enough
    if max_iter // n_split <= n_features:
        max_iter = n_features * n_split
        print(f"'max_iter' has been increased to {max_iter}")

    # use the cross-validation for define the best alpha of Lasso
    cv = KFold(n_splits=n_split, shuffle=True, random_state=seed)
    clf_lasso_cv = LassoCV(
        eps=eps, fit_intercept=False, cv=cv, tol=tol, max_iter=max_iter, n_jobs=n_jobs
    )
    # fit LassoCV
    clf_lasso_cv.fit(X_, y)
    
    # get coefficients and residuals
    beta_hat = clf_lasso_cv.coef_
    residual = clf_lasso_cv.predict(X_) - y

    # get the number of non-zero coefficients
    coef_max = np.max(np.abs(beta_hat))
    size_support = np.sum(np.abs(beta_hat) > tol * coef_max)
    # avoid dividing by 0
    size_support = min(size_support, n_samples - 1)

    #FIXME: citation
    # estimate the noise standard deviation (eq. 7 in [1])
    sigma_hat = norm(residual) / np.sqrt(n_samples - size_support)

    return sigma_hat, beta_hat


def group_reid(
    X,
    Y,
    fit_Y=True,
    stationary=True,
    method="simple",
    order=1,
    eps=1e-2,
    tol=1e-4,
    max_iter=10000,
    n_split=5,
    n_jobs=1,
    seed=0,
):
    """
    Estimation of the covariance matrix using group Reid procedure

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data.

    Y : ndarray, shape (n_samples, n_times)
        Target.

    fit_Y : bool, optional (default=True)
        If True, Y will be regressed against X by MultiTaskLassoCV
        and the covariance matrix is estimated on the residuals.
        Otherwise, covariance matrix is estimated directly on Y.

    stationary : bool, optional (default=True)
        If True, noise is considered to have the same magnitude for each
        time step. Otherwise, magnitude of the noise is not constant.

    method : str, optional (default='simple')
        If 'simple', the correlation matrix is estimated by taking the
        median of the correlation between two consecutive time steps
        and the noise standard deviation for each time step is estimated
        by taking the median of the standard deviations for every time step.
        In this case, the noise is considered to be stationary, i.e the 
        magnitude of the noise is constant for each time step.
        If 'AR', the order of the autoregressive (AR) model is given 
        by `order` and Yule-Walker method is used to estimate the 
        covariance matrix. In this case, the noise is considered to be 
        non-stationary, i.e. the magnitude of the noise is not constant.

    order : int, optional (default=1)
        If `stationary=True` and `method=AR`, `order` gives the order of the estimated 
        autoregressive model. `order` must be smaller than the number 
        of time steps.

    eps : float, optional (default=1e-2)
        Length of the cross-validation path.
        eps=1e-2 means that alpha_min / alpha_max = 1e-2.

    tol : float, optional (default=1e-4)
        The tolerance for the optimization: if the updates are smaller
        than `tol`, the optimization code checks the dual gap for optimality
        and continues until it is smaller than `tol`.

    max_iter : int, optional (default=1e4)
        The maximum number of iterations.
    
    n_split : int, optional (default=5)
        Number of splits in the KFold object used to cross-validate LassoCV.

    n_jobs : int or None, optional (default=1)
        Number of CPUs to use during the cross validation.

    seed: int, optional (default=0)
        Seed passed in the KFold object which is used to cross-validate
        LassoCV. This seed controls also the partitioning randomness.

    Returns
    -------
    cov_hat : ndarray, shape (n_times, n_times)
        Estimated covariance matrix.

    beta_hat : ndarray, shape (n_features, n_times)
        Estimated parameter matrix.

    References
    ----------
    .. [1] Chevalier, J. A., Gramfort, A., Salmon, J., & Thirion, B. (2020).
           Statistical control for spatio-temporal MEG/EEG source imaging with
           desparsified multi-task Lasso. In NeurIPS 2020-34h Conference on
           Neural Information Processing Systems.
    .. [2] Reid, S., Tibshirani, R., & Friedman, J. (2016). A study of error
            variance estimation in lasso regression. Statistica Sinica, 35-67.
    .. [3] Eshel, G. (2003). The yule walker equations for the AR coefficients. 
            Internet resource, 2, 68-73.
    """

    X_ = np.asarray(X)
    n_samples, n_features = X_.shape
    n_times = Y.shape[1]

    if method == "simple":
        print("Group reid: simple cov estimation")
    elif method == "AR":
        print(f"Group reid: {method}{order} cov estimation")
        if order > n_times - 1:
            raise ValueError(
                "The requested AR order is to high with "
                + "respect to the number of time steps."
            )
        elif not stationary:
            raise ValueError(
                "The AR method is not compatible with the non-stationary noise assumption."
            )
    else:
        raise ValueError("Unknown method for estimating the covariance matrix")

    if fit_Y:
        # check if max_iter is large enough
        if (max_iter // n_split) <= n_features:
            max_iter = n_features * n_split
            print(f"'max_iter' has been increased to {max_iter}")

        # use the cross-validation for define the best alpha of Lasso
        cv = KFold(n_splits=n_split, shuffle=True, random_state=seed)
        clf_mtlcv = MultiTaskLassoCV(
            eps=eps,
            fit_intercept=False,
            cv=cv,
            tol=tol,
            max_iter=max_iter,
            n_jobs=n_jobs,
        )
        # fit LassoCV
        clf_mtlcv.fit(X_, Y)
        
        # get coefficients and residuals
        beta_hat = clf_mtlcv.coef_
        residual = clf_mtlcv.predict(X_) - Y

        # get the number of non-zero coefficients
        row_max = np.max(np.sum(np.abs(beta_hat), axis=0))
        size_support = np.sum(np.sum(np.abs(beta_hat), axis=0) > tol * row_max)
        # avoid dividing by 0
        size_support = min(size_support, n_samples - 1)

    else:
        beta_hat = np.zeros((n_features, n_times))
        residual = np.copy(Y)
        size_support = 0

    #FIXME: citation
    # estimate the noise standard deviation (eq. 7 in [2])
    sigma_hat_raw = norm(residual, axis=0) / np.sqrt(n_samples - size_support)

    if stationary:
        #FIXME citation
        # consideration of stationary noise (section 2.5 of [2]) 
        sigma_hat = np.median(sigma_hat_raw) * np.ones(n_times)
        # compute rho from the empirical correlation matrix (section 2.5 of [2]) 
        corr_emp = np.corrcoef(residual.T)
    else:
        sigma_hat = sigma_hat_raw
        residual_rescaled = residual / sigma_hat
        corr_emp = np.corrcoef(residual_rescaled.T)

    #TODO: name of the method different than the name of the "function"
    # Median method
    if not stationary or method == "simple":
        rho_hat = np.median(np.diag(corr_emp, 1))
        #FIXME citation
        # estimate M (section 2.5 of [2]) 
        corr_hat = toeplitz(np.geomspace(1, rho_hat ** (n_times - 1), n_times))
        cov_hat = np.outer(sigma_hat, sigma_hat) * corr_hat

    #FIXME citation
    # Yule-Walker method (algorithm in section 3 of [3])
    elif stationary and method == "AR":
        # compute the autocorrelation coefficients of the AR model
        rho_ar = np.zeros(order + 1)
        rho_ar[0] = 1

        for i in range(1, order + 1):
            rho_ar[i] = np.median(np.diag(corr_emp, i))

        #FIXME citation
        # solve the Yule-Walker equations (see eq.2 in [3])
        R = toeplitz(rho_ar[:-1])
        coef_ar = solve(R, rho_ar[1:])

        # estimate the variance of the noise from the AR model
        residual_estimate = np.zeros((n_samples, n_times - order))
        for i in range(order):
            # time window used to estimate the residual from AR model
            start = order - i - 1
            end = -i - 1
            residual_estimate += coef_ar[i] * residual[:, start:end]
        residual_diff = residual[:, order:] - residual_estimate
        sigma_eps = np.median(norm(residual_diff, axis=0) / np.sqrt(n_samples))

        # estimation of the autocorrelation matrices
        rho_ar_full = np.zeros(n_times)
        rho_ar_full[: rho_ar.size] = rho_ar
        for i in range(order + 1, n_times):
            start = i - order
            end = i
            rho_ar_full[i] = np.dot(coef_ar[::-1], rho_ar_full[start:end])
        corr_hat = toeplitz(rho_ar_full)
        
        # estimation of the variance of an AR process
        # from wikipedia it should be  VAR(X_t)=\frac{\sigma_\epsilon^2}{1-\phi^2}
        # TODO there is a short difference in the code
        sigma_hat[:] = sigma_eps / np.sqrt((1 - np.dot(coef_ar, rho_ar[1:])))
        # estimation of the covariance based on the correlation matrix and sigma
        # COV(X_t, X_t) = COR(X_t, X_t) * \sigma^2 
        cov_hat = np.outer(sigma_hat, sigma_hat) * corr_hat

    return cov_hat, beta_hat


def empirical_snr(X, y, beta, noise=None):
    """
    Compute the signal noise ratio (SNR) for the linear model: y = X beta + noise

    Parameters
    ----------
    X : ndarray or scipy.sparse matrix, shape (n_samples, n_features)
        Data.

    y : ndarray, shape (n_samples,)
        Target.

    beta : ndarray, shape (n_features,)
        True parameter vector.

    noise : ndarray, shape (n_samples,), optional (default=None)
        True error vector.

    Returns
    -------
    snr_hat : float
        Empirical signal-to-noise ratio.
    """
    X = np.asarray(X)

    signal = np.dot(X, beta)

    if noise is None:
        noise = y - signal

    # compute signal-to-noise ratio
    snr_hat = np.var(signal) / np.var(noise)

    return snr_hat
