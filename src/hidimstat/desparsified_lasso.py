import warnings

import numpy as np
from joblib import Parallel, delayed
from numpy.linalg import multi_dot, norm
from scipy import stats
from scipy.linalg import inv, solve, toeplitz
from sklearn.base import check_is_fitted, clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Lasso, LassoCV, MultiTaskLasso, MultiTaskLassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_memory

from hidimstat._utils.docstring import _aggregate_docstring
from hidimstat._utils.regression import _alpha_max
from hidimstat._utils.utils import check_random_state, seed_estimator
from hidimstat.base_variable_importance import BaseVariableImportance
from hidimstat.statistical_tools.p_values import (
    pval_from_cb,
    pval_from_two_sided_pval_and_sign,
)


class DesparsifiedLasso(BaseVariableImportance):
    """
    Desparsified Lasso Estimator (also known as Debiased Lasso)

    Statistical inference in high-dimensional regression using the desparsified Lasso.
    Provides debiased coefficient estimates, confidence intervals and p-values.
    Algorithm based on Algorithm 1 of d-Lasso and d-MTLasso in
    :footcite:t:`chevalier2020statisticalthesis`.

    Parameters
    ----------
    model_y : LassoCV or MultiTaskLassoCV instance, default=LassoCV()
        Initial model for selecting relevant features. Must implement fit and predict.
        For single task use LassoCV, for multi-task use MultiTaskLassoCV.

    centered : bool, default=True
        Whether to center X and y before fitting.

    dof_ajdustement : bool, default=False
        Whether to apply degrees of freedom adjustment for small samples.

    model_x : Lasso or MultiTaskLasso instance, default=Lasso()
        Base model for nodewise regressions.

    alphas : array-like or None, default=None
        Regularization strengths for nodewise regressions. If None, computed from alpha_max_fraction.

    alpha_max_fraction : float, default=0.01
        Fraction of maximum alpha to use when alphas=None.

    random_state : int or RandomState, default=None
        Controls randomization.

    save_model_x : bool, default=False
        Whether to save fitted nodewise regression models.

    tolerance_reid : float, default=1e-4
        Convergence tolerance for noise estimation.

    noise_method : {'AR', 'median'}, default='AR'
        Method for noise covariance estimation:
        - 'AR': Autoregressive model
        - 'median': Median correlation

    order : int, default=1
        Order of AR model if noise_method='AR'.

    stationary : bool, default=True
        Whether to assume stationary noise.

    confidence : float, default=0.95
        Confidence level for intervals.

    distribution : str, default='norm'
        Distribution for p-values, only 'norm' supported.

    epsilon_pvalue : float, default=1e-14
        Small constant to avoid numerical issues.

    test : {'chi2', 'F'}, default='chi2'
        Test statistic for p-values:
        - 'chi2': Chi-squared test (large samples)
        - 'F': F-test (small samples)

    covariance : ndarray or None, default=None
        Pre-specified noise covariance matrix.

    n_jobs : int, default=1
        Number of parallel jobs.

    memory : str or Memory, default=None
        Cache for intermediate results.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    importances_ : ndarray of shape (n_features)
        Debiased coefficient estimates.

    pvalues_ : ndarray of shape (n_features)
        Two-sided p-values.

    pvalues_corr_ : ndarray of shape (n_features)
        Multiple testing corrected p-values.

    sigma_hat_ : float or ndarray of shape (n_task, n_task)
        Estimated noise level.

    precision_diagonal_ : ndarray of shape (n_features)
        Diagonal entries of precision matrix.

    confidence_bound_min_ : ndarray of shape (n_features)
        Lower confidence bounds.

    confidence_bound_max_ : ndarray of shape (n_features)
        Upper confidence bounds.
    """

    def __init__(
        self,
        model_y=LassoCV(
            eps=1e-2,
            fit_intercept=False,
            cv=KFold(n_splits=5),
            tol=1e-4,
            max_iter=5000,
            random_state=1,
            n_jobs=None,
        ),
        centered=True,
        dof_ajdustement=False,
        # parameters for model_x
        model_x=Lasso(max_iter=5000, tol=1e-3),
        alphas=None,
        alpha_max_fraction=0.01,
        random_state=None,
        save_model_x=False,
        # parameters for reid
        tolerance_reid=1e-4,
        noise_method="AR",
        order=1,
        stationary=True,
        # parameters for tests
        confidence=0.95,
        distribution="norm",
        epsilon_pvalue=1e-14,
        test="chi2",
        covariance=None,
        # parameters for optimization
        n_jobs=1,
        memory=None,
        verbose=0,
    ):
        super().__init__()
        if issubclass(LassoCV, model_y.__class__):
            self.n_task_ = 1
        elif issubclass(MultiTaskLassoCV, model_y.__class__):
            self.n_task_ = -1
        else:
            raise AssertionError("lasso_cv needs to be a LassoCV or a MultiTaskLassoCV")
        self.model_y = model_y
        self.centered = centered
        self.dof_ajdustement = dof_ajdustement
        # model x
        assert issubclass(Lasso, model_x.__class__) or issubclass(
            MultiTaskLasso, model_x.__class__
        ), "lasso needs to be a Lasso or a MultiTaskLasso"
        self.model_x = model_x
        self.alphas = alphas
        self.alpha_max_fraction = alpha_max_fraction
        self.save_model_x = save_model_x
        self.random_state = random_state
        # parameters for reid
        self.tolerance_reid = tolerance_reid
        self.noise_method = noise_method
        self.order = order
        self.stationary = stationary
        # parameters for test
        self.confidence = confidence
        self.distribution = distribution
        self.epsilon_pvalue = epsilon_pvalue
        self.covariance = covariance
        assert test == "chi2" or test == "F", f"Unknown test '{test}'"
        self.test = test
        # parameters for optimization
        self.n_jobs = n_jobs
        self.memory = memory
        self.verbose = verbose

        self.n_samples_ = None
        self.clf_ = None
        self.sigma_hat_ = None
        self.precision_diagonal_ = None
        self.confidence_bound_min_ = None
        self.confidence_bound_max_ = None
        self.pvalues_corr_ = None

    def fit(self, X, y):
        """
        Fit the Desparsified Lasso model.

        This method fits the Desparsified Lasso model to provide debiased coefficient estimates
        and statistical inference for high-dimensional regression.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data matrix.
        y : array-like of shape (n_samples,) or (n_samples, n_task)
            Target values. For single task, y should be 1D.
            For multi-task, y should be 2D with shape (n_samples, n_task).

        Returns
        -------
        self : object
            Returns the instance with fitted attributes:
            - `importances_` : Desparsified coefficient estimates
            - `sigma_hat_` : Estimated noise level
            - `precision_diagonal_` : Diagonal of precision matrix
            - `clf_` : Fitted nodewise regression models (if save_model_x=True)

        Notes
        -----
        The fitting process:
        1. Centers X and y if self.centered=True
        2. Fits initial Lasso using cross-validation
        3. Estimates noise variance using Reid method
        4. Computes nodewise Lasso regressions in parallel
        5. Calculates debiased coefficients and precision matrix
        """
        memory = check_memory(self.memory)
        rng = check_random_state(self.random_state)
        if self.n_task_ == -1:
            self.n_task_ = y.shape[1]

        # centering the data and the target variable
        if self.centered:
            X_ = StandardScaler(with_std=False).fit_transform(X)
            y_ = y - np.mean(y)
        else:
            X_ = X
            y_ = y
        self.n_samples_, n_features = X_.shape
        assert self.alphas is None or len(self.alphas) == n_features

        try:
            check_is_fitted(self.model_y)
        except NotFittedError:
            # check if max_iter is large enough
            if self.model_y.max_iter // self.model_y.cv.n_splits <= n_features:
                self.model_y.set_params(max_iter=n_features * self.model_y.cv.n_splits)
                warnings.warn(
                    f"'max_iter' has been increased to {self.model_y.max_iter}"
                )
            # use the cross-validation for define the best alpha of Lasso
            self.model_y.set_params(n_jobs=self.n_jobs)
            self.model_y.fit(X_, y_)

        # Lasso regression and noise standard deviation estimation
        self.sigma_hat_ = memory.cache(reid, ignore=["n_jobs"])(
            self.model_y.coef_,  # estimated support of the variable importance
            self.model_y.predict(X_) - y_,  # compute the residual,
            tolerance=self.tolerance_reid,
            # for group
            multioutput=self.n_task_ > 1,
            method=self.noise_method,
            order=self.order,
            stationary=self.stationary,
        )

        # define the alphas for the Nodewise Lasso
        if self.alphas is None:
            list_alpha_max = _alpha_max(X_, X_, fill_diagonal=True, axis=0)
            alphas = self.alpha_max_fraction * list_alpha_max
        gram = np.dot(X_.T, X_)  # Gram matrix

        # Calculating precision matrix (Nodewise Lasso)
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_joblib_compute_residuals)(
                X=X_,
                id_column=i,
                clf=seed_estimator(
                    clone(self.model_x).set_params(
                        alpha=alphas[i],
                    ),
                    random_state=rng_spwan,
                ),
                gram=gram,  # gram matrix is passed to the job to avoid memory issue
                return_clf=self.save_model_x,
            )
            for i, rng_spwan in enumerate(rng.spawn(n_features))
        )
        # Unpacking the results
        results = np.asarray(results, dtype=object)
        Z = np.stack(results[:, 0], axis=1)
        precision_diagonal = np.stack(results[:, 1])
        self.clf_ = [clf for clf in results[:, 2]]

        # Computing the degrees of freedom adjustment
        if self.dof_ajdustement:
            coefficient_max = np.max(np.abs(self.model_y.coef_))
            support = np.sum(np.abs(self.model_y.coef_) > 0.01 * coefficient_max)
            support = min(support, self.n_samples_ - 1)
            dof_factor = self.n_samples_ / (self.n_samples_ - support)
        else:
            dof_factor = 1

        # Computing Desparsified Lasso estimator and confidence intervals
        # Estimating the coefficient vector
        beta_bias = dof_factor * np.dot(y_.T, Z) / np.sum(X_ * Z, axis=0)

        # beta hat
        p = (np.dot(X_.T, Z) / np.sum(X_ * Z, axis=0)).T
        p_nodiagonal = p - np.diag(np.diag(p))
        p_nodiagonal = dof_factor * p_nodiagonal + (dof_factor - 1) * np.identity(
            n_features
        )
        self.importances_ = beta_bias.T - p_nodiagonal.dot(self.model_y.coef_.T)
        # confidence intervals
        self.precision_diagonal_ = precision_diagonal * dof_factor**2

        return self

    def _check_fit(self):
        """
        Check if the model has been fit properly.

        This method verifies that the model has been fitted by checking
        essential attributes (sigma_hat_ and lasso_cv).

        Raises
        ------
        ValueError
            If model hasn't been fit or required attributes are missing.
        """
        if (
            self.clf_ is None
            or self.importances_ is None
            or self.precision_diagonal_ is None
            or self.sigma_hat_ is None
        ):
            raise ValueError(
                "The Desparsified Lasso requires to be fit before any analysis"
            )

    def importance(self, X=None, y=None):
        """
        Compute desparsified lasso estimates, confidence intervals and p-values.

        Uses fitted model to calculate debiased coefficients along with confidence
        intervals and p-values. For single task regression, provides confidence
        intervals based on Gaussian approximation. For multi-task case,
        computes chi-squared or F test p-values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data matrix.
        y : array-like of shape (n_samples,) or (n_samples, n_task)
            Target values. For single task, y should be 1D or (n_samples, 1).
            For multi-task, y should be 2D with shape (n_samples, n_task).

        Returns
        -------
        importances_ : ndarray of shape (n_features,) or (n_features, n_task)
            Desparsified lasso coefficient estimates.

        Notes
        -----
        Updates several instance attributes:
        - `importances_`: Desparsified coefficient estimates
        - `pvalues_`: Two-sided p-values
        - `pvalues_corr_`: Multiple testing corrected p-values
        - `confidence_bound_min_`: Lower confidence bounds (single task only)
        - `confidence_bound_max_`: Upper confidence bounds (single task only)

        For multi-task case, p-values are based on chi-squared or F tests,
        configured by the test parameter ('chi2' or 'F').
        """
        if X is not None:
            warnings.warn("X won't be used.")
        if y is not None:
            warnings.warn("y won't be used.")
        self._check_fit()
        beta_hat = self.importances_

        if self.n_task_ == 1:
            # define the quantile for the confidence intervals
            quantile = stats.norm.ppf(1 - (1 - self.confidence) / 2)
            # see definition of lower and upper bound in algorithm 1
            # in `chevalier2020statisticalthesis`:
            # quantile_(1-alpha/2) * (n**(-1/2)) * sigma * (precision_diagonal**(1/2))
            confint_radius = np.abs(
                quantile
                * self.sigma_hat_
                * np.sqrt(self.precision_diagonal_)
                / np.sqrt(self.n_samples_)
            )
            self.confidence_bound_max_ = beta_hat + confint_radius
            self.confidence_bound_min_ = beta_hat - confint_radius

            pval, pval_corr, _, _ = pval_from_cb(
                self.confidence_bound_min_,
                self.confidence_bound_max_,
                confidence=self.confidence,
                distribution=self.distribution,
                eps=self.epsilon_pvalue,
            )
        else:
            covariance_hat = self.sigma_hat_
            if self.covariance is not None:
                covariance_hat = self.covariance
            theta_hat = self.n_samples_ * inv(covariance_hat)
            # Compute the two-sided p-values
            if self.test == "chi2":
                chi2_scores = (
                    np.diag(multi_dot([beta_hat, theta_hat, beta_hat.T]))
                    / self.precision_diagonal_
                )
                two_sided_pval = np.minimum(
                    2 * stats.chi2.sf(chi2_scores, df=self.n_task_), 1.0
                )
            elif self.test == "F":
                f_scores = (
                    np.diag(multi_dot([beta_hat, theta_hat, beta_hat.T]))
                    / self.precision_diagonal_
                    / self.n_task_
                )
                two_sided_pval = np.minimum(
                    2 * stats.f.sf(f_scores, dfd=self.n_samples_, dfn=self.n_task_),
                    1.0,
                )
            else:
                raise ValueError(f"Unknown test '{self.test}'")

            # Compute the p-values
            sign_beta = np.sign(np.sum(beta_hat, axis=1))
            pval, pval_corr, _, _ = pval_from_two_sided_pval_and_sign(
                two_sided_pval, sign_beta, eps=self.epsilon_pvalue
            )

        self.pvalues_ = pval
        self.pvalues_corr_ = pval_corr
        return self.importances_

    def fit_importance(self, X, y):
        """
        Fit and compute variable importance in one step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data matrix.
        y : array-like of shape (n_samples,) or (n_samples, n_task)
            Target values. For single task, y should be 1D or (n_samples, 1).
            For multi-task, y should be (n_samples, n_task).

        Returns
        -------
        importances_ : ndarray of shape (n_features,) or (n_features, n_task)
            Desparsified lasso coefficient estimates.
        """
        self.fit(X, y)
        return self.importance()


def _joblib_compute_residuals(X, id_column, clf, gram, return_clf):
    """
    Compute nodewise Lasso regression for desparsified Lasso estimation.

    For feature i, regresses X[:,i] against all other features to
    obtain residuals and precision matrix diagonal entry needed for debiasing.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix.
    id_column : int
        Index of feature to regress.
    clf : sklearn estimator
        Pre-configured estimator.
    return_clf : bool
        Whether to return fitted sklearn estimator model.

    Returns
    -------
    z : ndarray of shape (n_samples,)
        Residuals from regression.
    precision_diagonal : float
        Diagonal entry i of precision matrix estimate,
        computed as n * ||z_i||^2 / <x_i, z_i>^2.
    clf : sklearn estimator or None
        Fitted Lasso model if return_clf=True, else None.

    Notes
    -----
    Uses sklearn's Lasso with precomputed Gram matrix for efficiency.
    """

    n_samples, _ = X.shape

    # Removing the column to regress against the others
    X_minus_i = np.delete(X, id_column, axis=1)
    X_i = np.copy(X[:, id_column])

    clf.set_params(
        precompute=np.delete(np.delete(gram, id_column, axis=0), id_column, axis=1)
    )
    # Fitting the Lasso model and computing the residuals
    clf.fit(X_minus_i, X_i)
    z = X_i - clf.predict(X_minus_i)

    # Computing the diagonal of the covariance matrix,
    # which is used as an estimation of the noise covariance.
    precision_diagonal_i = n_samples * np.sum(z**2) / np.dot(X_i, z) ** 2

    if return_clf:
        return z, precision_diagonal_i, clf
    else:
        return z, precision_diagonal_i, None


def desparsified_lasso(
    X,
    y,
    model_y=LassoCV(
        eps=1e-2,
        fit_intercept=False,
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        tol=1e-3,
        max_iter=5000,
        random_state=0,
    ),
    centered=True,
    dof_ajdustement=False,
    # parameters for model_x
    model_x=Lasso(max_iter=5000, tol=1e-3),
    alphas=None,
    alpha_max_fraction=0.01,
    save_model_x=False,
    random_state=None,
    # parameters for reid
    tolerance_reid=1e-4,
    noise_method="AR",
    order=1,
    stationary=True,
    # parameters for tests
    confidence=0.95,
    distribution="norm",
    epsilon_pvalue=1e-14,
    test="chi2",
    covariance=None,
    # parameter for optimization
    n_jobs=1,
    memory=None,
    verbose=0,
    # parameter for selections
    k_lowest=None,
    percentile=None,
    threshold_min=None,
    threshold_max=None,
):
    methods = DesparsifiedLasso(
        model_y=model_y,
        centered=centered,
        dof_ajdustement=dof_ajdustement,
        model_x=model_x,
        alphas=alphas,
        alpha_max_fraction=alpha_max_fraction,
        save_model_x=save_model_x,
        random_state=random_state,
        tolerance_reid=tolerance_reid,
        noise_method=noise_method,
        order=order,
        stationary=stationary,
        confidence=confidence,
        distribution=distribution,
        epsilon_pvalue=epsilon_pvalue,
        test=test,
        covariance=covariance,
        n_jobs=n_jobs,
        memory=memory,
        verbose=verbose,
    )
    methods.fit_importance(X, y)
    selection = methods.pvalue_selection(
        k_lowest=k_lowest,
        percentile=percentile,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
    )
    return selection, methods.importances_, methods.pvalues_


# use the docstring of the class for the function
desparsified_lasso.__doc__ = _aggregate_docstring(
    [
        DesparsifiedLasso.__doc__,
        DesparsifiedLasso.__init__.__doc__,
        DesparsifiedLasso.fit_importance.__doc__,
        DesparsifiedLasso.pvalue_selection.__doc__,
    ],
    """
    Returns
    -------
    selection : ndarray of shape (n_features,)
        Boolean array indicating selected features (True = selected)
    importances : ndarray of shape (n_features,)
        Feature importance scores/test statistics. For features not selected 
        during screening, scores are set to 0.
    pvalues : ndarray of shape (n_features,)
        Two-sided p-values for each feature under Gaussian null hypothesis.
        For features not selected during screening, p-values are set to 1.
    """,
)


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
    Residual sum of squares based estimators for noise standard deviation
    estimation.

    This implementation follows the procedure described in
    :footcite:t:`fan2012variance` and :footcite:t:`reid2016study`.
    The beta_hat should correspond to the coefficient of Lasso with
    cross-validation, and the residual is based on this model.

    For group, the implementation is based on the procedure
    from :footcite:t:`chevalier2020statistical`.

    Parameters
    ----------
    beta_hat : ndarray, shape (n_features,) or (n_task, n_features)
        Estimated sparse coefficient vector from regression.

    residual : ndarray, shape (n_samples,) or (n_samples, n_task)
        Residuals from the regression model.

    tolerance : float, default=1e-4
        Threshold for considering coefficients as non-zero.

    multioutput : bool, default=False
        If True, handles multiple outputs (group case).

    stationary : bool, default=True
        Whether noise has constant magnitude across time steps.

    method : {'median', 'AR'}, (default='simple')
        Covariance estimation method:
        - 'median': Uses median correlation between consecutive time steps
        - 'AR': Uses Yule-Walker method with specified order

    order : int, default=1
        Order of AR model when method='AR'. Must be < n_task.

    Returns
    -------
    sigma_hat_raw or covariance_hat : float or ndarray
        For single output: estimated noise standard deviation
        For multiple outputs: estimated (n_task, n_task) covariance matrix

    Notes
    -----
    Implementation based on :footcite:t:`reid2016study` for single output
    and :footcite:t:`chevalier2020statistical` for multiple outputs.

    References
    ----------
    .. footbibliography::
    """
    if multioutput:
        n_task = beta_hat.shape[0]
    else:
        n_task = None

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
            if order > n_task - 1:
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
        ## compute empirical correlation of the residual
        if stationary:
            # consideration of stationary noise
            # (section 2.5 of `chevalier2020statistical`)
            sigma_hat = np.median(sigma_hat_raw) * np.ones(n_task)
            # compute rho from the empirical correlation matrix
            # (section 2.5 of `chevalier2020statistical`)
            correlation_empirical = np.corrcoef(residual.T)
        else:
            sigma_hat = sigma_hat_raw
            residual_rescaled = residual / sigma_hat
            correlation_empirical = np.corrcoef(residual_rescaled.T)

        covariance_hat = None
        # Median method
        if not stationary or method == "median":
            rho_hat = np.median(np.diag(correlation_empirical, 1))
            # estimate M (section 2.5 of `chevalier2020statistical`)
            correlation_hat = toeplitz(np.geomspace(1, rho_hat ** (n_task - 1), n_task))
            covariance_hat = np.outer(sigma_hat, sigma_hat) * correlation_hat

        # Yule-Walker method (algorithm in section 3 of `eshel2003yule`)
        elif stationary and method == "AR":
            # compute the autocorrelation coefficients of the AR model
            rho_ar = np.zeros(order + 1)
            rho_ar[0] = 1

            for i in range(1, order + 1):
                rho_ar[i] = np.median(np.diag(correlation_empirical, i))

            # solve the Yule-Walker equations (see eq.2 in `eshel2003yule`)
            R = toeplitz(rho_ar[:-1])
            coefficients_ar = solve(R, rho_ar[1:])

            # estimate the variance of the noise from the AR model
            residual_estimate = np.zeros((n_samples, n_task - order))
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
            rho_ar_full = np.zeros(n_task)
            rho_ar_full[: rho_ar.size] = rho_ar
            for i in range(order + 1, n_task):
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
        else:
            raise ValueError(
                f"Not support a combination of stationnary {stationary} and method {method}."
            )

        return covariance_hat
