import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve, toeplitz


def reid(
    beta_hat,
    residual,
    tolerance=1e-4,
    multioutput=False,
    stationary=True,
    method="median",
    order=1,
):
    """
    Residual sum of squares-based estimators for noise standard deviation
    estimation.

    This implementation follows the procedure described in
    :footcite:t:`fan2012variance` and :footcite:t:`reid2016study`.
    The beta_hat should correspond to the coefficient of Lasso with
    cross-validation, and the residual is based on this model.

    For groups, the implementation is based on the procedure
    from :footcite:t:`chevalier2020statistical`.

    Parameters
    ----------
    beta_hat : ndarray, shape (n_features,) or (n_times, n_features)
        Estimated sparse coefficient vector from regression.
    residual : ndarray, shape (n_samples,) or (n_samples, n_times)
        Residuals from the regression model.
    tolerance : float, default=1e-4
        Threshold for considering coefficients as non-zero.
    multioutput : bool, default=False
        If True, handles multiple outputs (group case).
    stationary : bool, default=True
        Whether noise has constant magnitude across time steps.
    method : {'median', 'AR'}, default='median'
        Method for covariance estimation in multioutput case:
        - 'median': Uses median correlation between consecutive time steps
        - 'AR': Uses Yule-Walker method with specified order
    order : int, default=1
        Order of AR model when method='AR'. Must be < n_times.

    Returns
    -------
    sigma_hat_raw or covariance_hat : float or ndarray
        For single output: estimated noise standard deviation
        For multiple outputs: estimated (n_times, n_times) covariance matrix

    Notes
    -----
    Implementation based on :footcite:t:`reid2016study` for single output
    and :footcite:t:`chevalier2020statistical` for multiple outputs.

    References
    ----------
    .. footbibliography::
    """
    if multioutput:
        n_times = beta_hat.shape[0]
    else:
        n_times = None
    n_samples = residual.shape[0]

    # get the number of non-zero coefficients
    # we consider that the coefficient with a value under
    # tolerance * coefficients_.max() is null
    coefficients_ = (
        np.sum(np.abs(beta_hat), axis=0)
        if len(beta_hat.shape) > 1
        else np.abs(beta_hat)
    )
    size_support = np.sum(coefficients_ > tolerance * coefficients_.max())

    # avoid dividing by 0
    size_support = min(size_support, n_samples - 1)

    # estimate the noise standard deviation (eq. 3 in `reid2016study`)
    sigma_hat_raw = norm(residual, axis=0) / np.sqrt(n_samples - size_support)

    if not multioutput:
        return sigma_hat_raw

    ## Computation of the covariance matrix for group
    else:
        if method == "median":
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
            correlation_emperical = np.corrcoef(residual.T)
        else:
            sigma_hat = sigma_hat_raw
            residual_rescaled = residual / sigma_hat
            correlation_emperical = np.corrcoef(residual_rescaled.T)

        # Median method
        if not stationary or method == "median":
            rho_hat = np.median(np.diag(correlation_emperical, 1))
            # estimate M (section 2.5 of `chevalier2020statistical`)
            correlation_hat = toeplitz(
                np.geomspace(1, rho_hat ** (n_times - 1), n_times)
            )
            covariance_hat = np.outer(sigma_hat, sigma_hat) * correlation_hat

        # Yule-Walker method (algorithm in section 3 of `eshel2003yule`)
        elif stationary and method == "AR":
            # compute the autocorrelation coefficients of the AR model
            rho_ar = np.zeros(order + 1)
            rho_ar[0] = 1

            for i in range(1, order + 1):
                rho_ar[i] = np.median(np.diag(correlation_emperical, i))

            # solve the Yule-Walker equations (see eq.2 in `eshel2003yule`)
            R = toeplitz(rho_ar[:-1])
            coefficients_ar = solve(R, rho_ar[1:])

            # estimate the variance of the noise from the AR model
            residual_estimate = np.zeros((n_samples, n_times - order))
            for i in range(order):
                # time window used to estimate the residual from AR model
                start = order - i - 1
                end = -i - 1
                residual_estimate += coefficients_ar[i] * residual[:, start:end]
            residual_difference = residual[:, order:] - residual_estimate
            sigma_epsilon = np.median(
                norm(residual_difference, axis=0) / np.sqrt(n_samples)
            )

            # estimation of the autocorrelation matrices
            rho_ar_full = np.zeros(n_times)
            rho_ar_full[: rho_ar.size] = rho_ar
            for i in range(order + 1, n_times):
                start = i - order
                end = i
                rho_ar_full[i] = np.dot(coefficients_ar[::-1], rho_ar_full[start:end])
            correlation_hat = toeplitz(rho_ar_full)

            # estimation of the variance of an AR process
            sigma_hat[:] = sigma_epsilon / np.sqrt(
                (1 - np.dot(coefficients_ar, rho_ar[1:]))
            )
            # estimation of the covariance based on the
            # correlation matrix and sigma
            # COV(X_t, X_t) = COR(X_t, X_t) * \sigma^2
            covariance_hat = np.outer(sigma_hat, sigma_hat) * correlation_hat

        return covariance_hat


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
    signal_noise_ratio_hat : float
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
    signal_noise_ratio_ = (np.linalg.norm(signal) / np.linalg.norm(noise)) ** 2

    return signal_noise_ratio_
