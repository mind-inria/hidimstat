import numpy as np
from joblib import Parallel, delayed
from numpy.linalg import multi_dot
from scipy import stats
from scipy.linalg import inv
from sklearn.linear_model import Lasso

from hidimstat.noise_std import reid
from hidimstat.stat_tools import pval_from_two_sided_pval_and_sign
from hidimstat.stat_tools import pval_from_cb


def desparsified_lasso(
    X,
    y,
    dof_ajdustement=False,
    max_iter=5000,
    tol=1e-3,
    alpha_max_fraction=0.01,
    eps=1e-2,
    tol_reid=1e-4,
    n_split=5,
    n_jobs=1,
    seed=0,
    verbose=0,
    group=False,
    cov=None,
    noise_method="AR",
    order=1,
    fit_Y=True,
    stationary=True,
):
    """
    Desparsified Lasso with confidence intervals

    Algorithm based on Algorithm 1 of d-Lasso and d-MTLasso in
    :cite:`chevalier2020statistical`.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data matrix.

    y : ndarray, shape (n_samples,) or (n_samples, n_times)
        Target vector for single response or matrix for multiple
        responses.

    dof_ajdustement : bool, optional (default=False)
        If True, applies degrees of freedom adjustment.
        If False, computes original Desparsified Lasso estimator.

    max_iter : int, optional (default=5000)
        Maximum iterations for Nodewise Lasso regressions.

    tol : float, optional (default=1e-3)
        Convergence tolerance for optimization.

    alpha_max_fraction : float, optional (default=0.01)
        Fraction of max lambda used for Lasso regularization.

    eps : float, optional (default=1e-2)
        Small constant used in noise estimation.

    tol_reid : float, optional (default=1e-4)
        Tolerance for Reid estimation.

    n_split : int, optional (default=5)
        Number of splits for cross-validation in Reid procedure.

    n_jobs : int, optional (default=1)
        Number of parallel jobs. Use -1 for all CPUs.

    seed : int, optional (default=0)
        Random seed for reproducibility.

    verbose : int, optional (default=0)
        Verbosity level for logging.

    group : bool, optional (default=False)
        If True, use group Lasso for multiple responses.

    cov : ndarray, shape (n_times, n_times), optional (default=None)
        Temporal covariance matrix of the noise.
        If None, it is estimated.

    noise_method : {'AR', 'simple'}, optional (default='AR')
        Method to estimate noise covariance:
        - 'simple': Uses median correlation between consecutive
                    timepoints
        - 'AR': Fits autoregressive model of specified order

    order : int, optional (default=1)
        Order of AR model when noise_method='AR'. Must be < n_times.

    fit_Y : bool, optional (default=True)
        Whether to fit Y in noise estimation.

    stationary : bool, optional (default=True)
        Whether to assume stationary noise in estimation.

    Returns
    -------
    beta_hat : ndarray, shape (n_features,) or (n_features, n_times)
        Desparsified Lasso coefficient estimates.

    sigma_hat/theta_hat : float or ndarray, shape (n_times, n_times)
        Estimated noise level (single response) or precision matrix
        (multiple responses).

    omega_diag : ndarray, shape (n_features,)
        Diagonal elements of the precision matrix.

    Notes
    -----
    The columns of `X` and `y` are always centered, this ensures that
    the intercepts of the Nodewise Lasso problems are all equal to zero
    and the intercept of the noise model is also equal to zero. Since
    the values of the intercepts are not of interest, the centering avoids
    the consideration of unecessary additional parameters.
    Also, you may consider to center and scale `X` beforehand, notably if
    the data contained in `X` has not been prescaled from measurements.

    References
    ----------
    .. footbibliography::
    """

    X_ = np.asarray(X)

    n_samples, n_features = X_.shape
    if group:
        n_times = y.shape[1]
        if cov is not None and cov.shape != (n_times, n_times):
            raise ValueError(
                f'Shape of "cov" should be ({n_times}, {n_times}),'
                + f' the shape of "cov" was ({cov.shape}) instead'
            )

    # centering the data and the target variable
    y_ = y - np.mean(y)
    X_ = X_ - np.mean(X_, axis=0)

    # Lasso regression and noise standard deviation estimation
    # TODO: other estimation of the noise standard deviation?
    sigma_hat, beta_reid = reid(
        X_,
        y_,
        eps=eps,
        tol=tol_reid,
        max_iter=max_iter,
        n_split=n_split,
        n_jobs=n_jobs,
        seed=seed,
        # for group
        group=group,
        method=noise_method,
        order=order,
        fit_Y=fit_Y,
        stationary=stationary,
    )

    # compute the Gram matrix
    gram = np.dot(X_.T, X_)
    gram_nodiag = np.copy(gram)
    np.fill_diagonal(gram_nodiag, 0)

    # define the alphas for the Nodewise Lasso
    # TODO why don't use the function _lambda_max instead of this?
    list_alpha_max = np.max(np.abs(gram_nodiag), axis=0) / n_samples
    alphas = alpha_max_fraction * list_alpha_max

    # Calculating precision matrix (Nodewise Lasso)
    Z, omega_diag = _compute_all_residuals(
        X_,
        alphas,
        gram,
        max_iter=max_iter,
        tol=tol,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # Computing the degrees of freedom adjustement
    if dof_ajdustement:
        coef_max = np.max(np.abs(beta_reid))
        support = np.sum(np.abs(beta_reid) > 0.01 * coef_max)
        support = min(support, n_samples - 1)
        dof_factor = n_samples / (n_samples - support)
    else:
        dof_factor = 1

    # Computing Desparsified Lasso estimator and confidence intervals
    # Estimating the coefficient vector
    beta_bias = dof_factor * np.dot(y_.T, Z) / np.sum(X_ * Z, axis=0)

    # beta hat
    P = (np.dot(X_.T, Z) / np.sum(X_ * Z, axis=0)).T
    P_nodiag = P - np.diag(np.diag(P))
    Id = np.identity(n_features)
    P_nodiag = dof_factor * P_nodiag + (dof_factor - 1) * Id
    beta_hat = beta_bias.T - P_nodiag.dot(beta_reid.T)
    # confidence intervals
    omega_diag = omega_diag * dof_factor**2

    if not group:
        return beta_hat, sigma_hat, omega_diag
    else:
        cov_hat = sigma_hat
        if cov is not None:
            cov_hat = cov
        theta_hat = n_samples * inv(cov_hat)
        return beta_hat, theta_hat, omega_diag


def desparsified_lasso_pvalue(
    n_samples,
    beta_hat,
    sigma_hat,
    omega_diag,
    confidence=0.95,
    distrib="norm",
    eps=1e-14,
    confidence_interval_only=False,
):
    """
    Calculate confidence intervals and p-values for desparsified lasso estimators.
    This function computes confidence intervals for the desparsified lasso
    estimator beta_hat.
    It can also return p-values derived from these confidence intervals.
    Parameters
    ----------
    n_samples : float
        The number of samples
    beta_hat : array-like
        The desparsified lasso coefficient estimates.
    sigma_hat : float
        Estimated noise level.
    omega_diag : array-like
        Diagonal elements of the precision matrix estimate.
    confidence : float, optional (default=0.95)
        Confidence level for intervals, must be in [0, 1].
    distrib : str, optional (default="norm")
        Distribution to use for p-value calculation.
        Currently only "norm" supported.
    eps : float, optional (default=1e-14)
        Small value to avoid numerical issues in p-value calculation.
    confidence_interval_only : bool, optional (default=False)
        If True, return only confidence intervals.
        If False, also return p-values.
    Returns
    -------
    If confidence_interval_only=True:
        cb_min : array-like
            Lower bounds of confidence intervals
        cb_max : array-like
            Upper bounds of confidence intervals
    If confidence_interval_only=False:
        pval : array-like
            P-values
        pval_corr : array-like
            Corrected p-values
        one_minus_pval : array-like
            1 - p-values
        one_minus_pval_corr : array-like
            1 - corrected p-values
        cb_min : array-like
            Lower bounds of confidence intervals
        cb_max : array-like
            Upper bounds of confidence intervals
    """
    # define the quantile for the confidence intervals
    quantile = stats.norm.ppf(1 - (1 - confidence) / 2)
    # TODO:why the double inverse of omega_diag?
    omega_invsqrt_diag = omega_diag ** (-0.5)
    confint_radius = np.abs(
        quantile * sigma_hat / (np.sqrt(n_samples) * omega_invsqrt_diag)
    )
    cb_max = beta_hat + confint_radius
    cb_min = beta_hat - confint_radius

    if confidence_interval_only:
        return cb_min, cb_max

    pval, pval_corr, one_minus_pval, one_minus_pval_corr = pval_from_cb(
        cb_min, cb_max, confidence=confidence, distrib=distrib, eps=eps
    )
    return pval, pval_corr, one_minus_pval, one_minus_pval_corr, cb_min, cb_max


def desparsified_group_lasso_pvalue(beta_hat, theta_hat, omega_diag, test="chi2"):
    """
    Compute p-values for the desparsified group Lasso estimator using
    chi-squared or F tests

    Parameters
    ----------
    beta_hat : ndarray, shape (n_features, n_times)
        Estimated parameter matrix from desparsified group Lasso.

    theta_hat : ndarray, shape (n_times, n_times)
        Estimated precision matrix (inverse covariance).

    omega_diag : ndarray, shape (n_features,)
        Diagonal elements of the precision matrix.

    test : {'chi2', 'F'}, optional (default='chi2')
        Statistical test for computing p-values:
        - 'chi2': Chi-squared test (recommended for large samples)
        - 'F': F-test (better for small samples)

    Returns
    -------
    pval : ndarray, shape (n_features,)
        Raw p-values, numerically accurate for positive effects
        (p-values close to 0).

    pval_corr : ndarray, shape (n_features,)
        P-values corrected for multiple testing using
        Benjamini-Hochberg procedure.

    one_minus_pval : ndarray, shape (n_features,)
        1 - p-values, numerically accurate for negative effects
        (p-values close to 1).

    one_minus_pval_corr : ndarray, shape (n_features,)
        1 - corrected p-values.

    Notes
    -----
    The chi-squared test assumes asymptotic normality while the F-test
    makes no such assumption and is preferable for small sample sizes.
    P-values are computed based on score statistics from the estimated
    coefficients and precision matrix.
    """
    n_features, n_times = beta_hat.shape
    n_samples = omega_diag.shape[0]

    # Compute the two-sided p-values
    if test == "chi2":
        chi2_scores = np.diag(multi_dot([beta_hat, theta_hat, beta_hat.T])) / omega_diag
        two_sided_pval = np.minimum(2 * stats.chi2.sf(chi2_scores, df=n_times), 1.0)
    elif test == "F":
        f_scores = (
            np.diag(multi_dot([beta_hat, theta_hat, beta_hat.T])) / omega_diag / n_times
        )
        two_sided_pval = np.minimum(
            2 * stats.f.sf(f_scores, dfd=n_samples, dfn=n_times), 1.0
        )
    else:
        raise ValueError(f"Unknown test '{test}'")

    # Compute the p-values
    sign_beta = np.sign(np.sum(beta_hat, axis=1))
    pval, pval_corr, one_minus_pval, one_minus_pval_corr = (
        pval_from_two_sided_pval_and_sign(two_sided_pval, sign_beta)
    )

    return pval, pval_corr, one_minus_pval, one_minus_pval_corr


def _compute_all_residuals(
    X, alphas, gram, max_iter=5000, tol=1e-3, n_jobs=1, verbose=0
):
    """
    Nodewise Lasso for computing residuals and precision matrix diagonal.

    For each feature, fits a Lasso regression against all other features
    to estimate the precision matrix and residuals needed for the
    desparsified Lasso estimator.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data matrix.

    alphas : ndarray, shape (n_features,)
        Lasso regularization parameters, one per feature.

    gram : ndarray, shape (n_features, n_features)
        Precomputed Gram matrix X.T @ X to speed up computations.

    max_iter : int, optional (default=5000)
        Maximum number of iterations for Lasso optimization.

    tol : float, optional (default=1e-3)
        Convergence tolerance for Lasso optimization.

    n_jobs : int or None, optional (default=1)
        Number of parallel jobs. None means using all processors.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting the models:
        0 = silent
        1 = progress bar
        >1 = more detailed output

    Returns
    -------
    Z : ndarray, shape (n_samples, n_features)
        Matrix of residuals from nodewise regressions.

    omega_diag : ndarray, shape (n_features,)
        Diagonal entries of the precision matrix estimate.

    Notes
    -----
    This implements the nodewise Lasso procedure from :cite:`chevalier2020statistical`
    for estimating entries of the precision matrix needed in the
    desparsified Lasso. The procedure regresses each feature against
    all others using Lasso to obtain residuals and precision matrix estimates.

    References
    ----------
    .. footbibliography::
    """

    n_samples, n_features = X.shape

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_compute_residuals)(
            X=X,
            column_index=i,
            alpha=alphas[i],
            gram=gram,
            max_iter=max_iter,
            tol=tol,
        )
        for i in range(n_features)
    )

    # Unpacking the results
    results = np.asarray(results, dtype=object)
    Z = np.stack(results[:, 0], axis=1)
    omega_diag = np.stack(results[:, 1])

    return Z, omega_diag


def _compute_residuals(X, column_index, alpha, gram, max_iter=5000, tol=1e-3):
    """
    Compute nodewise Lasso regression for desparsified Lasso estimation

    For feature i, regresses X[:,i] against all other features to
    obtain residuals and precision matrix diagonal entry needed for debiasing.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Centered input data matrix

    column_index : int
        Index i of feature to regress

    alpha : float
        Lasso regularization parameter

    gram : ndarray, shape (n_features, n_features)
        Precomputed X.T @ X matrix

    max_iter : int, default=5000
        Maximum Lasso iterations

    tol : float, default=1e-3
        Optimization tolerance

    Returns
    -------
    z : ndarray, shape (n_samples,)
        Residuals from regression

    omega_diag_i : float
        Diagonal entry i of precision matrix estimate,
        computed as n * ||z||^2 / <x_i, z>^2

    Notes
    -----
    Uses sklearn's Lasso with precomputed Gram matrix for efficiency.
    """

    n_samples, n_features = X.shape
    i = column_index

    # Removing the column to regress against the others
    X_new = np.delete(X, i, axis=1)
    y_new = np.copy(X[:, i])

    # Method used for computing the residuals of the Nodewise Lasso.
    # here we use the Lasso method
    gram_ = np.delete(np.delete(gram, i, axis=0), i, axis=1)
    clf = Lasso(alpha=alpha, precompute=gram_, max_iter=max_iter, tol=tol)

    # Fitting the Lasso model and computing the residuals
    clf.fit(X_new, y_new)
    z = y_new - clf.predict(X_new)

    # Computing the diagonal of the covariance matrix
    omega_diag_i = n_samples * np.sum(z**2) / np.dot(y_new, z) ** 2

    return z, omega_diag_i
