import warnings

import numpy as np
from joblib import Parallel, delayed
from numpy.linalg import multi_dot
from scipy import stats
from scipy.linalg import inv
from sklearn.base import check_is_fitted, clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Lasso, LassoCV, MultiTaskLassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_memory

from hidimstat._utils.docstring import _aggregate_docstring
from hidimstat._utils.regression import _alpha_max
from hidimstat._utils.utils import check_random_state, seed_estimator
from hidimstat.base_variable_importance import BaseVariableImportance
from hidimstat.noise_std import reid
from hidimstat.statistical_tools.p_values import (
    pval_from_cb,
    pval_from_two_sided_pval_and_sign,
)


class DesparsifiedLasso(BaseVariableImportance):
    """
    Desparsified Lasso

    Algorithm based on Algorithm 1 of d-Lasso and d-MTLasso in
    :footcite:t:`chevalier2020statisticalthesis`.

    Parameters
    ----------
    model_y : LassoCV or MultiTaskLassoCV instance, default=LassoCV()
        CV object used for initial Lasso fit.

    model_x : Lasso instance, default=Lasso()
        Base Lasso estimator used for nodewise regressions.

    centered : bool, default=True
        Whether to center X and y.

    dof_ajdustement : bool, default=False
        If True, applies degrees of freedom adjustment from :footcite:t:`bellec2022biasing`.

    alpha_max_fraction : float, default=0.01
        Fraction of max alpha used for nodewise Lasso regularization.

    tolerance_reid : float, default=1e-4
        Tolerance for Reid variance estimation method.

    random_state : int, RandomState instance or None, default=None
        Controls randomization in CV splitter and Lasso fits.

    covariance : ndarray of shape (n_times, n_times) or None, default=None
        Temporal noise covariance matrix. If None, estimated from data.

    noise_method : {'AR', 'median'}, default='AR'
        Method to estimate noise covariance:
        - 'AR': Autoregressive model
        - 'median': Median correlation between consecutive timepoints

    order : int, default=1
        Order of AR model when noise_method='AR'. Must be < n_times.

    stationary : bool, default=True
        Whether to assume stationary noise in estimation.

    confidence : float, default=0.95
        Confidence level for intervals, must be in [0, 1].

    distribution : str, default='norm'
        Distribution for p-value calculation. Only 'norm' supported.

    epsilon_pvalue : float, default=1e-14
        Small value to avoid numerical issues in p-values.

    test : {'chi2', 'F'}, default='chi2'
        Test for p-values:
        - 'chi2': Chi-squared test (large samples)
        - 'F': F-test (small samples)

    n_jobs : int, default=1
        Number of parallel jobs. -1 means all CPUs.

    memory : str or Memory object, default=None
        Used to cache nodewise Lasso computations.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    importances_ : ndarray of shape (n_features,) or (n_features, n_times)
        Desparsified Lasso coefficient estimates.

    pvalues_ : ndarray of shape (n_features,)
        Two-sided p-values.

    pvalues_corr_ : ndarray of shape (n_features,)
        Multiple testing corrected p-values.

    sigma_hat_ : float or ndarray of shape (n_times, n_times)
        Estimated noise level or precision matrix.

    confidence_bound_min_ : ndarray of shape (n_features,)
        Lower confidence bounds.

    confidence_bound_max_ : ndarray of shape (n_features,)
        Upper confidence bounds.

    Notes
    -----
    X and y are always centered. Consider pre-scaling X if not already scaled.
    Chi-squared test assumes asymptotic normality, F-test preferred for small samples.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        model_y=LassoCV(
            eps=1e-2,
            fit_intercept=False,
            cv=KFold(n_splits=5, shuffle=True, random_state=0),
            tol=1e-4,
            max_iter=5000,
            random_state=1,
            n_jobs=1,
        ),
        model_x=Lasso(max_iter=5000, tol=1e-3),
        centered=True,
        dof_ajdustement=False,
        alpha_max_fraction=0.01,
        tolerance_reid=1e-4,
        random_state=None,
        covariance=None,
        noise_method="AR",
        order=1,
        stationary=True,
        confidence=0.95,
        distribution="norm",
        epsilon_pvalue=1e-14,
        test="chi2",
        save_model_x=False,
        n_jobs=1,
        memory=None,
        verbose=0,
    ):
        super().__init__()
        assert issubclass(
            Lasso, model_x.__class__
        ), "lasso needs to be a Lasso or a MultiTaskLasso"
        self.model_x = model_x
        if issubclass(LassoCV, model_y.__class__):
            self.n_times_ = 1
        elif issubclass(MultiTaskLassoCV, model_y.__class__):
            self.n_times_ = -1
        else:
            raise AssertionError("lasso_cv needs to be a LassoCV or a MultiTaskLassoCV")
        self.model_y = model_y
        self.centered = centered
        self.dof_ajdustement = dof_ajdustement
        self.alpha_max_fraction = alpha_max_fraction
        self.tolerance_reid = tolerance_reid
        self.covariance = covariance
        self.noise_method = noise_method
        self.order = order
        self.stationary = stationary
        self.confidence = confidence
        self.distribution = distribution
        self.epsilon_pvalue = epsilon_pvalue
        assert test == "chi2" or test == "F", f"Unknown test '{test}'"
        self.test = test
        self.save_model_x = save_model_x
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.memory = memory
        self.verbose = verbose

        self.sigma_hat_ = None
        self.confidence_bound_min_ = None
        self.confidence_bound_max_ = None
        self.pvalues_corr_ = None
        self.precision_diagonal_ = None
        self.clf_ = None
        self.n_samples_ = None

    def fit(self, X, y):
        """
        Fit the Desparsified Lasso model.

        This method fits the Desparsified Lasso model, which provides debiased estimates
        and statistical inference for high-dimensional linear models through a two-step
        procedure involving initial Lasso estimation followed by bias correction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data matrix.
        y : array-like of shape (n_samples,) or (n_samples, n_times)
            Target values. For single task, y should be 1D or (n_samples, 1).
            For multi-task, y should be (n_samples, n_times).

        Returns
        -------
        self : object
            Returns the fitted instance.

        Notes
        -----
        Main steps:
        1. Optional data centering
        2. Initial Lasso fit using cross-validation
        3. Computation of residuals
        4. Estimation of noise standard deviation
        5. Preparation for subsequent importance score calculation
        """
        memory = check_memory(self.memory)
        rng = check_random_state(self.random_state)
        if self.n_times_ == -1:
            self.n_times_ = y.shape[1]

        # centering the data and the target variable
        if self.centered:
            X_ = StandardScaler(with_std=False).fit_transform(X)
            y_ = y - np.mean(y)
        else:
            X_ = X
            y_ = y
        self.n_samples_, n_features = X_.shape

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
            multioutput=self.n_times_ > 1,
            method=self.noise_method,
            order=self.order,
            stationary=self.stationary,
        )

        # define the alphas for the Nodewise Lasso
        list_alpha_max = _alpha_max(X_, X_, fill_diagonal=True, axis=0)
        alphas = self.alpha_max_fraction * list_alpha_max
        gram = np.dot(X_.T, X_)  # Gram matrix

        # Calculating precision matrix (Nodewise Lasso)
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_compute_residuals)(
                X=X_,
                id_column=i,
                clf=seed_estimator(
                    clone(self.model_x).set_params(
                        alpha=alphas[i],
                        precompute=np.delete(np.delete(gram, i, axis=0), i, axis=1),
                    ),
                    random_state=rng_spwan,
                ),
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
        if self.sigma_hat_ is None:
            raise ValueError(
                "The Desparsified Lasso requires to be fit before any analysis"
            )
        try:
            check_is_fitted(self.model_y)
        except NotFittedError:
            raise ValueError(
                "The Desparsified Lasso requires to be fit before any analysis"
            )

    def importance(self, X, y):
        """
        Compute desparsified lasso estimates and confidence intervals.

        Calculates debiased coefficients, confidence intervals and p-values
        using the desparsified lasso method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data matrix.
        y : array-like of shape (n_samples,) or (n_samples, n_times)
            Target values. For single task, y should be 1D or (n_samples, 1).
            For multi-task, y should be (n_samples, n_times).

        Returns
        -------
        importances_ : ndarray of shape (n_features,) or (n_features, n_times)
            Desparsified lasso coefficient estimates.

        Attributes
        ----------
        importances_ : same as return value
        pvalues_ : ndarray of shape (n_features,)
            Two-sided p-values for each feature.
        pvalues_corr_ : ndarray of shape (n_features,)
            Multiple testing corrected p-values.
        confidence_bound_min_ : ndarray of shape (n_features,)
            Lower confidence bounds (only for single task).
        confidence_bound_max_ : ndarray of shape (n_features,)
            Upper confidence bounds (only for single task).

        Notes
        -----
        The method:
        1. Performs nodewise lasso regressions to estimate precision matrix
        2. Debiases initial lasso estimates
        3. Computes confidence intervals and p-values
        4. For multi-task case, uses chi-squared or F test
        """
        self._check_fit()
        beta_hat = self.importances_

        if self.n_times_ == 1:
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

            pval, pval_corr, one_minus_pval, one_minus_pval_corr = pval_from_cb(
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
                    2 * stats.chi2.sf(chi2_scores, df=self.n_times_), 1.0
                )
            elif self.test == "F":
                f_scores = (
                    np.diag(multi_dot([beta_hat, theta_hat, beta_hat.T]))
                    / self.precision_diagonal_
                    / self.n_times_
                )
                two_sided_pval = np.minimum(
                    2 * stats.f.sf(f_scores, dfd=self.n_samples_, dfn=self.n_times_),
                    1.0,
                )
            else:
                raise ValueError(f"Unknown test '{self.test}'")

            # Compute the p-values
            sign_beta = np.sign(np.sum(beta_hat, axis=1))
            pval, pval_corr, one_minus_pval, one_minus_pval_corr = (
                pval_from_two_sided_pval_and_sign(two_sided_pval, sign_beta)
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
        y : array-like of shape (n_samples,) or (n_samples, n_times)
            Target values. For single task, y should be 1D or (n_samples, 1).
            For multi-task, y should be (n_samples, n_times).

        Returns
        -------
        importances_ : ndarray of shape (n_features,) or (n_features, n_times)
            Desparsified lasso coefficient estimates.
        """
        self.fit(X, y)
        return self.importance(X, y)


def _compute_residuals(X, id_column, clf, return_clf):
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
    random_state : Generator, default=None
        Random state for reproducibility

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

    n_samples, _ = X.shape

    # Removing the column to regress against the others
    X_minus_i = np.delete(X, id_column, axis=1)
    X_i = np.copy(X[:, id_column])

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
    model_x=Lasso(max_iter=5000, tol=1e-3),
    centered=True,
    dof_ajdustement=False,
    alpha_max_fraction=0.01,
    tolerance_reid=1e-4,
    random_state=None,
    covariance=None,
    noise_method="AR",
    order=1,
    stationary=True,
    confidence=0.95,
    distribution="norm",
    epsilon_pvalue=1e-14,
    test="chi2",
    n_jobs=1,
    memory=None,
    verbose=0,
    k_lowest=None,
    percentile=None,
    threshold_min=None,
    threshold_max=None,
):
    methods = DesparsifiedLasso(
        model_y=model_y,
        model_x=model_x,
        centered=centered,
        dof_ajdustement=dof_ajdustement,
        alpha_max_fraction=alpha_max_fraction,
        tolerance_reid=tolerance_reid,
        random_state=random_state,
        covariance=covariance,
        noise_method=noise_method,
        order=order,
        stationary=stationary,
        confidence=confidence,
        distribution=distribution,
        epsilon_pvalue=epsilon_pvalue,
        test=test,
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
