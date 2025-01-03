import numpy as np
from joblib import Parallel, delayed
from numpy.linalg import multi_dot
from scipy import stats
from scipy.linalg import inv
from sklearn.linear_model import Lasso

from hidimstat.noise_std import group_reid, reid
from hidimstat.stat_tools import pval_from_two_sided_pval_and_sign


def desparsified_lasso(
    X,
    y,
    dof_ajdustement=False,
    confidence=0.95,
    max_iter=5000,
    tol=1e-3,
    alpha_max_fraction=0.01,
    n_jobs=1,
    verbose=0,
):
    """
    Desparsified Lasso with confidence intervals
    
    FIXME: fix citation of the algorithm
    Algorithm based Algo 1 of d-Lasso in [6]

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data.

    y : ndarray, shape (n_samples,)
        Target.

    dof_ajdustement : bool, optional (default=False)
        If True, makes the degrees of freedom adjustement (cf. [4]_ and [5]_).
        Otherwise, the original Desparsified Lasso estimator is computed
        (cf. [1]_ and [2]_ and [3]_).

    confidence : float, optional (default=0.95)
        Confidence level used to compute the confidence intervals.
        Each value should be in the range [0, 1].

    max_iter : int, optional (default=5000)
        The maximum number of iterations when regressing, by Lasso,
        each column of the design matrix against the others.

    tol : float, optional (default=1e-3)
        The tolerance for the optimization of the Lasso problems: if the
        updates are smaller than `tol`, the optimization code checks the
        dual gap for optimality and continues until it is smaller than `tol`.

    alpha_max_fraction : float, optional (default=0.01)
        Only used if method='lasso'.
        Then alpha = alpha_max_fraction * alpha_max.

    n_jobs : int or None, optional (default=1)
        Number of CPUs to use during the Nodewise Lasso.

    verbose: int, optional (default=1)
        The verbosity level: if non zero, progress messages are printed
        when computing the Nodewise Lasso in parralel.
        The frequency of the messages increases with the verbosity level.

    Returns
    -------
    beta_hat : array, shape (n_features,)
        Estimated parameter vector.

    cb_min : array, shape (n_features)
        Lower bound of the confidence intervals on the parameter vector.

    cb_max : array, shape (n_features)
        Upper bound of the confidence intervals on the parameter vector.

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
    .. [1] Zhang, C. H., & Zhang, S. S. (2014). Confidence intervals for
           low dimensional parameters in high dimensional linear models.
           Journal of the Royal Statistical Society: Series B: Statistical
           Methodology, 217-242.

    .. [2] Van de Geer, S., Bühlmann, P., Ritov, Y. A., & Dezeure, R. (2014).
           On asymptotically optimal confidence regions and tests for
           high-dimensional models. Annals of Statistics, 42(3), 1166-1202.

    .. [3] Javanmard, A., & Montanari, A. (2014). Confidence intervals and
           hypothesis testing for high-dimensional regression. The Journal
           of Machine Learning Research, 15(1), 2869-2909.

    .. [4] Bellec, P. C., & Zhang, C. H. (2019). De-biasing the lasso with
           degrees-of-freedom adjustment. arXiv preprint arXiv:1902.08885.

    .. [5] Celentano, M., Montanari, A., & Wei, Y. (2020). The Lasso with
           general Gaussian designs with applications to hypothesis testing.
           arXiv preprint arXiv:2007.13716.
    .. [6] Chevalier, Jérôme-Alexis. “Statistical Control of Sparse Models in High Dimension.” 
        Phdthesis, Université Paris-Saclay, 2020. https://theses.hal.science/tel-03147200.

    """

    X_ = np.asarray(X)

    n_samples, n_features = X_.shape

    # centering the data and the target variable
    y_ = y - np.mean(y)
    X_ = X_ - np.mean(X_, axis=0)

    # define the quantile for the confidence intervals
    quantile = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    # Lasso regression and noise standard deviation estimation
    #TODO: other estimation of the noise standard deviation?
    #      or add all the parameters of reid in the function?
    sigma_hat, beta_lasso = reid(X_, y_, n_jobs=n_jobs)

    # compute the Gram matrix
    gram = np.dot(X_.T, X_)
    gram_nodiag = np.copy(gram)
    np.fill_diagonal(gram_nodiag, 0)

    # define the alphas for the Nodewise Lasso
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
        coef_max = np.max(np.abs(beta_lasso))
        support = np.sum(np.abs(beta_lasso) > 0.01 * coef_max)
        support = min(support, n_samples - 1)
        dof_factor = n_samples / (n_samples - support)
    else:
        dof_factor = 1

    # Computing Desparsified Lasso estimator and confidence intervals
    # Estimating the coefficient vector
    beta_bias = dof_factor * np.dot(y_.T, Z) / np.sum(X_ * Z, axis=0)

    # beta hat
    P = ((Z.T.dot(X_)).T / np.sum(X_ * Z, axis=0)).T
    P_nodiag = P - np.diag(np.diag(P))
    Id = np.identity(n_features)
    P_nodiag = dof_factor * P_nodiag + (dof_factor - 1) * Id
    beta_hat = beta_bias - P_nodiag.dot(beta_lasso)

    # confidence intervals
    omega_diag = omega_diag * dof_factor ** 2
    #TODO:why the double inverse of omega_diag?
    omega_invsqrt_diag = omega_diag ** (-0.5)
    confint_radius = np.abs(
        quantile * sigma_hat / (np.sqrt(n_samples) * omega_invsqrt_diag) 
    )
    cb_max = beta_hat + confint_radius
    cb_min = beta_hat - confint_radius

    return beta_hat, cb_min, cb_max


def desparsified_group_lasso(
    X,
    Y,
    cov=None,
    max_iter=5000,
    tol=1e-3,
    alpha_max_fraction=0.01,
    noise_method="AR",
    order=1,
    n_jobs=1,
    memory=None,
    verbose=0,
):
    """
    Desparsified Group Lasso

    Algorithm based Algorithm 1 of d-MTLasso in [1]
    

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data.

    Y : ndarray, shape (n_samples, n_times)
        Target.

    cov : ndarray, shape (n_times, n_times), optional (default=None)
        If None, a temporal covariance matrix of the noise is estimated.
        Otherwise, `cov` is the temporal covariance matrix of the noise.

    max_iter : int, optional (default=5000)
        The maximum number of iterations when regressing, by Lasso,
        each column of the design matrix against the others.

    tol : float, optional (default=1e-3)
        The tolerance for the optimization of the Lasso problems: if the
        updates are smaller than `tol`, the optimization code checks the
        dual gap for optimality and continues until it is smaller than `tol`.

    alpha_max_fraction : float, optional (default=0.01)
        Only used if method='lasso'.
        Then alpha = alpha_max_fraction * alpha_max.

    noise_method : str, optional (default='simple')
        If 'simple', the correlation matrix is estimated by taking the
        median of the correlation between two consecutive time steps
        and the noise standard deviation for each time step is estimated
        by taking the median of the standard deviations for every time step.
        If 'AR', the order of the AR model is given by `order` and
        Yule-Walker method is used to estimate the covariance matrix.

    order : int, optional (default=1)
        If `method=AR`, `order` gives the order of the estimated autoregressive
        model. `order` must be smaller than the number of time steps.

    n_jobs : int or None, optional (default=1)
        Number of CPUs to use during the Nodewise Lasso.

    verbose: int, optional (default=1)
        The verbosity level: if non zero, progress messages are printed
        when computing the Nodewise Lasso in parralel.
        The frequency of the messages increases with the verbosity level.

    Returns
    -------
    beta_hat : ndarray, shape (n_features, n_times)
        Estimated parameter matrix.

    theta_hat : ndarray, shape (n_times, n_times)
        Estimated precision matrix.
        
    omega_diag : ndarray, shape (n_features,)
        Diagonal of the covariance matrix.
        
    Notes
    -----
    The columns of `X` and the matrix `Y` are always centered, this ensures
    that the intercepts of the Nodewise Lasso problems are all equal to zero
    and the intercept of the noise model is also equal to zero. Since
    the values of the intercepts are not of interest, the centering avoids
    the consideration of unecessary additional parameters.
    Also, you may consider to center and scale `X` beforehand, notably if
    the data contained in `X` has not been prescaled from measurements.

    References
    ----------
    .. [1] Chevalier, J. A., Gramfort, A., Salmon, J., & Thirion, B. (2020).
           Statistical control for spatio-temporal MEG/EEG source imaging with
           desparsified multi-task Lasso. In NeurIPS 2020-34h Conference on
           Neural Information Processing Systems.
    """

    X_ = np.asarray(X)

    n_samples, n_features = X_.shape
    n_times = Y.shape[1]

    if cov is not None and cov.shape != (n_times, n_times):
        raise ValueError(
            f'Shape of "cov" should be ({n_times}, {n_times}),'
            + f' the shape of "cov" was ({cov.shape}) instead'
        )

    # centering the data and the target variable
    Y_ = Y - np.mean(Y)
    X_ = X_ - np.mean(X_, axis=0)

    
    # Lasso regression and noise standard deviation estimation
    #TODO: other estimation of the noise standard deviation?
    #      or add all the parameters of group reid in the function?
    cov_hat, beta_mtl = group_reid(
        X, Y, method=noise_method, order=order, n_jobs=n_jobs
    )
    if cov is not None:
        cov_hat = cov
    theta_hat = n_samples * inv(cov_hat)

    # compute the Gram matrix
    gram = np.dot(X_.T, X_)
    gram_nodiag = np.copy(gram)
    np.fill_diagonal(gram_nodiag, 0)

    # define the alphas for the Nodewise Lasso
    list_alpha_max = np.max(np.abs(gram_nodiag), axis=0) / n_samples
    alphas = alpha_max_fraction * list_alpha_max

    # Calculating precision matrix (Nodewise Lasso)
    Z, omega_diag = _compute_all_residuals(
        X,
        alphas,
        gram,
        max_iter=max_iter,
        tol=tol,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # Computing Desparsified Lasso estimator and confidence intervals
    # Estimating the coefficient vector
    beta_bias = Y_.T.dot(Z) / np.sum(X_ * Z, axis=0)

    # beta hat
    P = (np.dot(X.T, Z) / np.sum(X * Z, axis=0)).T
    P_nodiag = P - np.diag(np.diag(P))
    beta_hat = beta_bias.T - P_nodiag.dot(beta_mtl.T)

    return beta_hat, theta_hat, omega_diag


def desparsified_group_lasso_pvalue(beta_hat, theta_hat, omega_diag, test="chi2"):
    """
    Compute p-values for the desparsified group Lasso estimator
    
    Parameters
    ----------
    beta_hat : ndarray, shape (n_features, n_times)
        Estimated parameter matrix.
        
    theta_hat : ndarray, shape (n_times, n_times)
        Estimated precision matrix.
        
    omega_diag : ndarray, shape (n_features,)
        Diagonal of the covariance matrix.
        
    test : str, optional (default='chi2')
        Statistical test used to compute p-values. 'chi2' corresponds
        to a chi-squared test and 'F' corresponds to an F-test.
        
    Returns
    -------
    pval : ndarray, shape (n_features,)
        p-value, with numerically accurate values for
        positive effects (ie., for p-value close to zero).
        
    pval_corr : ndarray, shape (n_features,)
        p-value corrected for multiple testing.
        
    one_minus_pval : ndarray, shape (n_features,)
        One minus the p-value, with numerically accurate values
        for negative effects (ie., for p-value close to one).
        
    one_minus_pval_corr : ndarray, shape (n_features,)
        One minus the p-value corrected for multiple testing.
    """
    n_features, n_times = beta_hat.shape
    n_samples = omega_diag.shape[0]
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
    
    sign_beta = np.sign(np.sum(beta_hat, axis=1))
    pval, pval_corr, one_minus_pval, one_minus_pval_corr = (
        pval_from_two_sided_pval_and_sign(two_sided_pval, sign_beta)
    )
    return pval, pval_corr, one_minus_pval, one_minus_pval_corr

def _compute_all_residuals(
    X, alphas, gram, max_iter=5000, tol=1e-3, n_jobs=1, verbose=0
):
    """Nodewise Lasso. Compute all the residuals: regressing each column of the
    design matrix against the other columns"""

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

    results = np.asarray(results, dtype=object)
    Z = np.stack(results[:, 0], axis=1)
    omega_diag = np.stack(results[:, 1])

    return Z, omega_diag


def _compute_residuals(
    X, column_index, alpha, gram, max_iter=5000, tol=1e-3
):
    """Compute the residuals of the regression of a given column of the
    design matrix against the other columns"""

    n_samples, n_features = X.shape
    i = column_index

    X_new = np.delete(X, i, axis=1)
    y = np.copy(X[:, i])

    # Method used for computing the residuals of the Nodewise Lasso.
    # here we use the Lasso method
    gram_ = np.delete(np.delete(gram, i, axis=0), i, axis=1)
    clf = Lasso(alpha=alpha, precompute=gram_, max_iter=max_iter, tol=tol)

    clf.fit(X_new, y)
    z = y - clf.predict(X_new)

    omega_diag_i = n_samples * np.sum(z**2) / np.dot(y, z) ** 2

    return z, omega_diag_i

