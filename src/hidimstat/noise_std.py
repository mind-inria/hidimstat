import numpy as np
from numpy.linalg import norm
from scipy.linalg import toeplitz, solve
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
from sklearn.model_selection import KFold


def reid(X, y, eps=1e-2, tol=1e-4, max_iter=1e4, n_jobs=1, seed=0):
    """Estimation of noise standard deviation using Reid procedure

    Parameters
    -----------
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
    """

    X = np.asarray(X)
    n_samples, n_features = X.shape

    if max_iter // 5 <= n_features:
        max_iter = n_features * 5
        print(f"'max_iter' has been increased to {max_iter}")

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    clf_lasso_cv = \
        LassoCV(eps=eps, normalize=False, fit_intercept=False,
                cv=cv, tol=tol, max_iter=max_iter, n_jobs=n_jobs)

    clf_lasso_cv.fit(X, y)
    beta_hat = clf_lasso_cv.coef_
    residual = clf_lasso_cv.predict(X) - y
    coef_max = np.max(np.abs(beta_hat))
    support = np.sum(np.abs(beta_hat) > tol * coef_max)

    # avoid dividing by 0
    support = min(support, n_samples - 1)

    sigma_hat = norm(residual) / np.sqrt(n_samples - support)

    return sigma_hat, beta_hat


def group_reid(X, Y, fit_Y=True, stationary=True, method='simple', order=1,
               eps=1e-2, tol=1e-4, max_iter=1e4, n_jobs=1, seed=0):

    """Estimation of the covariance matrix using group Reid procedure

    Parameters
    -----------
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
        If 'AR', the order of the AR model is given by `order` and
        Yule-Walker method is used to estimate the covariance matrix.

    order : int, optional (default=1)
        If `stationary=True` and `method=AR`, `order` gives the
        order of the estimated autoregressive model. `order` must
        be smaller than the number of time steps.

    eps : float, optional (default=1e-2)
        Length of the cross-validation path.
        eps=1e-2 means that alpha_min / alpha_max = 1e-2.

    tol : float, optional (default=1e-4)
        The tolerance for the optimization: if the updates are smaller
        than `tol`, the optimization code checks the dual gap for optimality
        and continues until it is smaller than `tol`.

    max_iter : int, optional (default=1e4)
        The maximum number of iterations.

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
    """

    X = np.asarray(X)
    n_samples, n_features = X.shape
    n_times = Y.shape[1]

    if method == 'simple':
        print('Group reid: simple cov estimation')
    else:
        print(f'Group reid: {method}{order} cov estimation')

    if (max_iter // 5) <= n_features:
        max_iter = n_features * 5
        print(f"'max_iter' has been increased to {max_iter}")

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    if fit_Y:

        clf_mtlcv = \
            MultiTaskLassoCV(eps=eps, normalize=False, fit_intercept=False,
                             cv=cv, tol=tol, max_iter=max_iter, n_jobs=n_jobs)

        clf_mtlcv.fit(X, Y)
        beta_hat = clf_mtlcv.coef_
        residual = clf_mtlcv.predict(X) - Y
        row_max = np.max(np.sum(np.abs(beta_hat), axis=0))
        support = np.sum(np.sum(np.abs(beta_hat), axis=0) > tol * row_max)

        # avoid dividing by 0
        support = min(support, n_samples - 1)

    else:

        beta_hat = np.zeros((n_features, n_times))
        residual = np.copy(Y)
        support = 0

    sigma_hat_raw = norm(residual, axis=0) / np.sqrt(n_samples - support)

    if stationary:
        sigma_hat = np.median(sigma_hat_raw) * np.ones(n_times)
        corr_emp = np.corrcoef(residual.T)
    else:
        sigma_hat = sigma_hat_raw
        residual_rescaled = residual / sigma_hat
        corr_emp = np.corrcoef(residual_rescaled.T)

    # Median method
    if not stationary or method == 'simple':

        rho_hat = np.median(np.diag(corr_emp, 1))
        corr_hat = \
            toeplitz(np.geomspace(1, rho_hat ** (n_times - 1), n_times))
        cov_hat = np.outer(sigma_hat, sigma_hat) * corr_hat

    # Yule-Walker method
    elif stationary and method == 'AR':

        if order > n_times - 1:
            raise ValueError('The requested AR order is to high with ' +
                             'respect to the number of time steps.')

        rho_ar = np.zeros(order + 1)
        rho_ar[0] = 1

        for i in range(1, order + 1):
            rho_ar[i] = np.median(np.diag(corr_emp, i))

        A = toeplitz(rho_ar[:-1])
        coef_ar = solve(A, rho_ar[1:])

        residual_estimate = np.zeros((n_samples, n_times - order))

        for i in range(order):
            # time window used to estimate the residual from AR model
            start = order - i - 1
            end = - i - 1
            residual_estimate += coef_ar[i] * residual[:, start:end]

        residual_diff = residual[:, order:] - residual_estimate
        sigma_eps = np.median(norm(residual_diff, axis=0) / np.sqrt(n_samples))

        rho_ar_full = np.zeros(n_times)
        rho_ar_full[:rho_ar.size] = rho_ar

        for i in range(order + 1, n_times):
            start = i - order
            end = i
            rho_ar_full[i] = np.dot(coef_ar[::-1], rho_ar_full[start:end])

        corr_hat = toeplitz(rho_ar_full)
        sigma_hat[:] = sigma_eps / np.sqrt((1 - np.dot(coef_ar, rho_ar[1:])))
        cov_hat = np.outer(sigma_hat, sigma_hat) * corr_hat

    else:
        raise ValueError('Unknown method for estimating the covariance matrix')

    return cov_hat, beta_hat


def empirical_snr(X, y, beta, noise=None):
    """Compute the SNR for the linear model: y = X beta + noise

    Parameters
    -----------
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

    sig_signal = np.linalg.norm(signal - np.mean(signal))
    sig_noise = np.linalg.norm(noise - np.mean(noise))
    snr_hat = (sig_signal / sig_noise) ** 2

    return snr_hat
