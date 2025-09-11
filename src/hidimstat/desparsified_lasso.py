import numpy as np
from joblib import Parallel, delayed
from numpy.linalg import multi_dot
from scipy import stats
from scipy.linalg import inv
from sklearn.linear_model import Lasso
from sklearn.utils.validation import check_memory

from hidimstat.noise_std import reid
from hidimstat.statistical_tools.p_values import (
    pval_from_two_sided_pval_and_sign,
    pval_from_cb,
)
from hidimstat._utils.regression import _alpha_max
from hidimstat._utils.utils import get_seed_generator, check_random_state


def desparsified_lasso(
    X,
    y,
    dof_ajdustement=False,
    max_iteration=5000,
    tolerance=1e-3,
    alpha_max_fraction=0.01,
    epsilon=1e-2,
    tolerance_reid=1e-4,
    n_splits=5,
    n_jobs=1,
    random_state=None,
    memory=None,
    verbose=0,
    multioutput=False,
    covariance=None,
    noise_method="AR",
    order=1,
    stationary=True,
):
    """
    Desparsified Lasso

    Algorithm based on Algorithm 1 of d-Lasso and d-MTLasso in
    :footcite:t:`chevalier2020statisticalthesis`.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data matrix.

    y : ndarray, shape (n_samples,) or (n_samples, n_times)
        Target vector for single response or matrix for multiple
        responses.

    dof_ajdustement : bool, optional (default=False)
        If True, applies degrees of freedom adjustment from :footcite:t:`bellec2022biasing`.
        If False, computes original Desparsified Lasso estimator.

    max_iteration : int, optional (default=5000)
        Maximum iterations for Nodewise Lasso regressions.

    tolerance : float, optional (default=1e-3)
        Convergence tolerance for optimization.

    alpha_max_fraction : float, optional (default=0.01)
        Fraction of max alpha used for Lasso regularization.

    epsilon : float, optional (default=1e-2)
        Small constant used in noise estimation.

    tolerance_reid : float, optional (default=1e-4)
        Tolerance for variance estimation with the Reid method.

    n_splits : int, optional (default=5)
        Number of splits for cross-validation in Reid procedure.

    n_jobs : int, optional (default=1)
        Number of parallel jobs. Use -1 for all CPUs.

    random_state : int, default=None
        Random seed for reproducibility.

    memory : str or joblib.Memory object, optional (default=None)
        Used to cache the output of the computation of the Nodewise Lasso.
        By default, no caching is done. If a string is given, it is the path
        to the caching directory.

    verbose : int, default=0
        Verbosity level for logging.

    multioutput : bool, default=False
        If True, use group Lasso for multiple responses.

    covariance : ndarray, shape (n_times, n_times), default=None
        Temporal covariance matrix of the noise.
        If None, it is estimated.

    noise_method : {'AR', 'median'}, default='AR'
        Method to estimate noise covariance:
        - 'median': Uses median correlation between consecutive
        timepoints
        - 'AR': Fits autoregressive model of specified order

    order : int, default=1
        Order of AR model when noise_method='AR'. Must be < n_times.

    stationary : bool, default=True
        Whether to assume stationary noise in estimation.

    Returns
    -------
    beta_hat : ndarray, shape (n_features,) or (n_features, n_times)
        Desparsified Lasso coefficient estimates.

    sigma_hat/theta_hat : float or ndarray, shape (n_times, n_times)
        Estimated noise level (single response) or precision matrix
        (multiple responses).

    precision_diagonal : ndarray, shape (n_features,)
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
    Other relevant references: :footcite:t:`van2014asymptotically`,
    :footcite:t:`zhang2014confidence`.

    References
    ----------
    .. footbibliography::
    """
    memory = check_memory(memory)
    rng = check_random_state(random_state)

    X_ = np.asarray(X)

    n_samples, n_features = X_.shape
    if multioutput:
        n_times = y.shape[1]
        if covariance is not None and covariance.shape != (n_times, n_times):
            raise ValueError(
                f'Shape of "cov" should be ({n_times}, {n_times}),'
                + f' the shape of "cov" was ({covariance.shape}) instead'
            )

    # centering the data and the target variable
    y_ = y - np.mean(y)
    X_ = X_ - np.mean(X_, axis=0)

    # Lasso regression and noise standard deviation estimation
    sigma_hat, beta_reid = memory.cache(reid, ignore=["n_jobs"])(
        X_,
        y_,
        epsilon=epsilon,
        tolerance=tolerance_reid,
        max_iterance=max_iteration,
        n_splits=n_splits,
        n_jobs=n_jobs,
        random_state=rng,
        # for group
        multioutput=multioutput,
        method=noise_method,
        order=order,
        stationary=stationary,
    )

    # define the alphas for the Nodewise Lasso
    list_alpha_max = _alpha_max(X_, X_, fill_diagonal=True, axis=0)
    alphas = alpha_max_fraction * list_alpha_max

    # Calculating precision matrix (Nodewise Lasso)
    Z, precision_diagonal = memory.cache(
        _compute_all_residuals, ignore=["n_jobs", "verbose"]
    )(
        X_,
        alphas,
        np.dot(X_.T, X_),  # Gram matrix
        max_iteration=max_iteration,
        tolerance=tolerance,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=rng,
    )

    # Computing the degrees of freedom adjustement
    if dof_ajdustement:
        coefficient_max = np.max(np.abs(beta_reid))
        support = np.sum(np.abs(beta_reid) > 0.01 * coefficient_max)
        support = min(support, n_samples - 1)
        dof_factor = n_samples / (n_samples - support)
    else:
        dof_factor = 1

    # Computing Desparsified Lasso estimator and confidence intervals
    # Estimating the coefficient vector
    beta_bias = dof_factor * np.dot(y_.T, Z) / np.sum(X_ * Z, axis=0)

    # beta hat
    P = (np.dot(X_.T, Z) / np.sum(X_ * Z, axis=0)).T
    P_nodiagonal = P - np.diag(np.diag(P))
    Id = np.identity(n_features)
    P_nodiagonal = dof_factor * P_nodiagonal + (dof_factor - 1) * Id
    beta_hat = beta_bias.T - P_nodiagonal.dot(beta_reid.T)
    # confidence intervals
    precision_diagonal = precision_diagonal * dof_factor**2

    if not multioutput:
        return beta_hat, sigma_hat, precision_diagonal
    else:
        covariance_hat = sigma_hat
        if covariance is not None:
            covariance_hat = covariance
        theta_hat = n_samples * inv(covariance_hat)
        return beta_hat, theta_hat, precision_diagonal


def desparsified_lasso_pvalue(
    n_samples,
    beta_hat,
    sigma_hat,
    precision_diagonal,
    confidence=0.95,
    distribution="norm",
    epsilon=1e-14,
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
    beta_hat : ndarray, shape (n_features,)
        The desparsified lasso coefficient estimates.
    sigma_hat : float
        Estimated noise level.
    precision_diagonal : ndarray, shape (n_features,)
        Diagonal elements of the precision matrix estimate.
    confidence : float, default=0.95
        Confidence level for intervals, must be in [0, 1].
    distribution : str, default="norm"
        Distribution to use for p-value calculation.
        Currently only "norm" supported.
    epsilon : float, default=1e-14
        Small value to avoid numerical issues in p-value calculation.

    Returns
    -------
    pval : ndarray, shape (n_features,)
        P-values
    pval_corr : ndarray, shape (n_features,)
        Corrected p-values
    one_minus_pval : ndarray, shape (n_features,)
        1 - p-values
    one_minus_pval_corr : ndarray, shape (n_features,)
        1 - corrected p-values
    confidence_bound_min : ndarray, shape (n_features,)
        Lower bounds of confidence intervals
    confidence_bound_max : ndarray, shape (n_features,)
        Upper bounds of confidence intervals
    """
    # define the quantile for the confidence intervals
    quantile = stats.norm.ppf(1 - (1 - confidence) / 2)
    # see definition of lower and upper bound in algorithm 1
    # in `chevalier2020statisticalthesis`:
    # quantile_(1-alpha/2) * (n**(-1/2)) * sigma * (precision_diagonal**(1/2))
    confint_radius = np.abs(
        quantile * sigma_hat * np.sqrt(precision_diagonal) / np.sqrt(n_samples)
    )
    confidence_bound_max = beta_hat + confint_radius
    confidence_bound_min = beta_hat - confint_radius

    pval, pval_corr, one_minus_pval, one_minus_pval_corr = pval_from_cb(
        confidence_bound_min,
        confidence_bound_max,
        confidence=confidence,
        distribution=distribution,
        eps=epsilon,
    )
    return (
        pval,
        pval_corr,
        one_minus_pval,
        one_minus_pval_corr,
        confidence_bound_min,
        confidence_bound_max,
    )


def desparsified_group_lasso_pvalue(
    beta_hat, theta_hat, precision_diagonal, test="chi2"
):
    """
    Compute p-values for the desparsified group Lasso estimator using
    chi-squared or F tests

    Parameters
    ----------
    beta_hat : ndarray, shape (n_features, n_times)
        Estimated parameter matrix from desparsified group Lasso.

    theta_hat : ndarray, shape (n_times, n_times)
        Estimated precision matrix (inverse covariance).

    precision_diagonal : ndarray, shape (n_features,)
        Diagonal elements of the precision matrix.

    test : {'chi2', 'F'}, default='chi2'
        Statistical test for computing p-values:
        - 'chi2': Chi-squared test (recommended for large samples)
        - 'F': F-test

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
    The Chi-squared test assumes asymptotic normality while the F-test
    is preferable for small sample sizes.
    P-values are computed based on score statistics from the estimated
    coefficients and precision matrix.
    """
    n_features, n_times = beta_hat.shape
    n_samples = precision_diagonal.shape[0]

    # Compute the two-sided p-values
    if test == "chi2":
        chi2_scores = (
            np.diag(multi_dot([beta_hat, theta_hat, beta_hat.T])) / precision_diagonal
        )
        two_sided_pval = np.minimum(2 * stats.chi2.sf(chi2_scores, df=n_times), 1.0)
    elif test == "F":
        f_scores = (
            np.diag(multi_dot([beta_hat, theta_hat, beta_hat.T]))
            / precision_diagonal
            / n_times
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
    X,
    alphas,
    gram,
    max_iteration=5000,
    tolerance=1e-3,
    n_jobs=1,
    verbose=0,
    random_state=None,
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

    max_itereration : int, optional (default=5000)
        Maximum number of iterations for Lasso optimization.

    tolerance : float, optional (default=1e-3)
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

    precision_diagonal : ndarray, shape (n_features,)
        Diagonal entries of the precision matrix estimate.

    Notes
    -----
    This implements the nodewise Lasso procedure from :footcite:t:`chevalier2020statisticalthesis`
    for estimating entries of the precision matrix needed in the
    desparsified Lasso. The procedure regresses each feature against
    all others using Lasso to obtain residuals and precision matrix estimates.

    References
    ----------
    .. footbibliography::
    """

    n_samples, n_features = X.shape

    generator_seeds = get_seed_generator(random_state)
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_compute_residuals)(
            X=X,
            id_column=i,
            alpha=alphas[i],
            gram=gram,
            max_iteration=max_iteration,
            tolerance=tolerance,
            random_state=generator_seeds.get_seed(i),
        )
        for i in range(n_features)
    )

    # Unpacking the results
    results = np.asarray(results, dtype=object)
    Z = np.stack(results[:, 0], axis=1)
    precision_diagonal = np.stack(results[:, 1])

    return Z, precision_diagonal


def _compute_residuals(
    X, id_column, alpha, gram, max_iteration=5000, tolerance=1e-3, random_state=None
):
    """
    Compute nodewise Lasso regression for desparsified Lasso estimation

    For feature i, regresses X[:,i] against all other features to
    obtain residuals and precision matrix diagonal entry needed for debiasing.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Centered input data matrix

    id_column : int
        Index i of feature to regress

    alpha : float
        Lasso regularization parameter

    gram : ndarray, shape (n_features, n_features)
        Precomputed X.T @ X matrix

    max_iteration : int, default=5000
        Maximum Lasso iterations

    tolerance : float, default=1e-3
        Optimization tolerance

    Returns
    -------
    z : ndarray, shape (n_samples,)
        Residuals from regression

    precision_diagonal_i : float
        Diagonal entry i of precision matrix estimate,
        computed as n * ||z||^2 / <x_i, z>^2

    Notes
    -----
    Uses sklearn's Lasso with precomputed Gram matrix for efficiency.
    """
    random_state = check_random_state(random_state)
    n_samples, n_features = X.shape

    # Removing the column to regress against the others
    X_minus_i = np.delete(X, id_column, axis=1)
    X_i = np.copy(X[:, id_column])

    # Method used for computing the residuals of the Nodewise Lasso.
    # here we use the Lasso method
    gram_ = np.delete(np.delete(gram, id_column, axis=0), id_column, axis=1)
    clf = Lasso(
        alpha=alpha,
        precompute=gram_,
        max_iter=max_iteration,
        tol=tolerance,
        random_state=random_state,
    )

    # Fitting the Lasso model and computing the residuals
    clf.fit(X_minus_i, X_i)
    z = X_i - clf.predict(X_minus_i)

    # Computing the diagonal of the covariance matrix,
    # which is used as an estimation of the noise covariance.
    precision_diagonal_i = n_samples * np.sum(z**2) / np.dot(X_i, z) ** 2

    return z, precision_diagonal_i
