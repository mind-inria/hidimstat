import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve, toeplitz
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
from sklearn.model_selection import KFold


def reid(
    X,
    y,
    eps=1e-2,
    tol=1e-4,
    max_iter=10000,
    n_split=5,
    n_jobs=1,
    seed=0,
    group=False,
    fit_Y=True,
    stationary=True,
    method="simple",
    order=1,
):
    """
    Residual sum of squares based estimators for noise standard deviation
    estimation.

    This implementation follows the procedure described in
    :footcite:`fan2012variance` and :cite:`reid2016study`. It uses Lasso with
    cross-validation to estimate both the noise standard deviation and model
    coefficients.

    For group, the implementation is based on the procedure
    from :footcite:`chevalier2020statistical`.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data matrix.

    y : ndarray, shape (n_samples,)/(n_samples, n_times)
        Target vector. The time means the presence of groups.

    eps : float, optional (default=1e-2)
        Length of the cross-validation path, where alpha_min / alpha_max = eps.
        Smaller values create a finer grid.

    tol : float, optional (default=1e-4)
        Tolerance for optimization convergence. The algorithm stops
        when updates are smaller than tol and dual gap is smaller than tol.

    max_iter : int, optional (default=10000)
        Maximum number of iterations for the optimization algorithm.

    n_split : int, optional (default=5)
        Number of folds for cross-validation.

    n_jobs : int, optional (default=1)
        Number of parallel jobs for cross-validation.
        -1 means using all processors.

    seed : int, optional (default=0)
        Random seed for reproducible cross-validation splits.

    fit_Y : bool, (default=True)
        Whether to use MultiTaskLassoCV to fit Y against X.
        If False, covariance is estimated directly from Y.

    stationary : bool, (default=True)
        Whether noise has constant magnitude across time steps.

    method : {'simple', 'AR'}, (default='simple')
        Covariance estimation method:
        - 'simple': Uses median correlation between consecutive time steps
        - 'AR': Uses Yule-Walker method with specified order

    order : int, default=1
        Order of AR model when method='AR'. Must be < n_times.

    Returns
    -------
    sigma_hat/cov_hat : float/ndarray, shape (n_times, n_times)
        Estimated noise standard deviation based on residuals
        or estimated covariance matrix for group.

    beta_hat : ndarray, shape (n_features,)/(n_features, n_times)
        Estimated sparse coefficient vector from Lasso regression.

    References
    ----------
    .. footbibliography::
    """

    X_ = np.asarray(X)
    n_samples, n_features = X_.shape
    if group:
        n_times = y.shape[1]
    if fit_Y:
        # check if max_iter is large enough
        if max_iter // n_split <= n_features:
            max_iter = n_features * n_split
            print(f"'max_iter' has been increased to {max_iter}")

        # use the cross-validation for define the best alpha of Lasso
        cv = KFold(n_splits=n_split, shuffle=True, random_state=seed)
        Refit_CV = MultiTaskLassoCV if group else LassoCV
        clf_cv = Refit_CV(
            eps=eps,
            fit_intercept=False,
            cv=cv,
            tol=tol,
            max_iter=max_iter,
            n_jobs=n_jobs,
        )
        clf_cv.fit(X_, y)

        # Estimate the support of the variable importance
        beta_hat = clf_cv.coef_
        residual = clf_cv.predict(X_) - y

        # get the number of non-zero coefficients
        coef_ = np.sum(np.abs(beta_hat), axis=0) if len(beta_hat.shape)> 1 else np.abs(beta_hat)
        size_support = np.sum(coef_ > tol * coef_.max())

        # avoid dividing by 0
        size_support = min(size_support, n_samples - 1)
    else:
        # null model
        # TODO Why do we need a null model?
        beta_hat = np.zeros((n_features, n_times))
        residual = np.copy(y)
        size_support = 0

    # estimate the noise standard deviation (eq. 7 in `fan2012variance`)
    sigma_hat_raw = norm(residual, axis=0) / np.sqrt(n_samples - size_support)

    if not group:
        return sigma_hat_raw, beta_hat

    ## Computation of the covariance matrix for group
    else:
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
                    "The AR method is not compatible with the non-stationary"
                    + " noise assumption."
                )
        else:
            raise ValueError("Unknown method for estimating the covariance matrix")
        ## compute emperical correlation of the residual
        if stationary:
            # consideration of stationary noise
            # (section 2.5 of `chevalier2020statistical`)
            sigma_hat = np.median(sigma_hat_raw) * np.ones(n_times)
            # compute rho from the empirical correlation matrix
            # (section 2.5 of `chevalier2020statistical`)
            corr_emp = np.corrcoef(residual.T)
        else:
            sigma_hat = sigma_hat_raw
            residual_rescaled = residual / sigma_hat
            corr_emp = np.corrcoef(residual_rescaled.T)

        # TODO: Why the name of the method is different
        #           than the name of the "function"?
        # Median method
        if not stationary or method == "simple":
            rho_hat = np.median(np.diag(corr_emp, 1))
            # estimate M (section 2.5 of `chevalier2020statistical`)
            corr_hat = toeplitz(np.geomspace(1, rho_hat ** (n_times - 1), n_times))
            cov_hat = np.outer(sigma_hat, sigma_hat) * corr_hat

        # Yule-Walker method (algorithm in section 3 of `eshel2003yule`)
        elif stationary and method == "AR":
            # compute the autocorrelation coefficients of the AR model
            rho_ar = np.zeros(order + 1)
            rho_ar[0] = 1

            for i in range(1, order + 1):
                rho_ar[i] = np.median(np.diag(corr_emp, i))

            # solve the Yule-Walker equations (see eq.2 in `eshel2003yule`)
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
            # from wikipedia it should be:
            # VAR(X_t)=\frac{\sigma_\epsilon^2}{1-\phi^2}
            # TODO there is a short difference between the code
            # and the above formula
            sigma_hat[:] = sigma_eps / np.sqrt((1 - np.dot(coef_ar, rho_ar[1:])))
            # estimation of the covariance based on the
            # correlation matrix and sigma
            # COV(X_t, X_t) = COR(X_t, X_t) * \sigma^2
            cov_hat = np.outer(sigma_hat, sigma_hat) * corr_hat

        return cov_hat, beta_hat


def empirical_snr(X, y, beta, noise=None):
    """
    Compute the empirical signal-to-noise ratio (SNR) for
    the linear model y = X @ beta + noise.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix.

    y : ndarray, shape (n_samples,)
        Target vector.

    beta : ndarray, shape (n_features,)
        Parameter vector.

    noise : ndarray, shape (n_samples,), optional
        Noise vector. If None, computed as y - X @ beta.

    Returns
    -------
    snr_hat : float
        Empirical SNR computed as var(signal) / var(noise).

    Notes
    -----
    SNR measures the ratio of signal power to noise power,
    indicating model estimation quality.
    Higher values suggest better signal recovery.
    """
    X = np.asarray(X)

    signal = np.dot(X, beta)

    if noise is None:
        noise = y - signal

    # compute signal-to-noise ratio
    snr_hat = np.var(signal) / np.var(noise)

    return snr_hat
