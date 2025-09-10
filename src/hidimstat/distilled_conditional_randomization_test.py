import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler

from hidimstat._utils.docstring import _aggregate_docstring
from hidimstat._utils.regression import _alpha_max
from hidimstat._utils.utils import _check_vim_predict_method
from hidimstat.base_variable_importance import BaseVariableImportance


class D0CRT(BaseVariableImportance):
    """
    Implements distilled conditional randomization test (dCRT) without interactions.

    This class provides a fast implementation of the Conditional Randomization
    Test :footcite:t:`candes2018panning` using the distillation process
    from :footcite:t:`liu2022fast`. The approach accelerates variable selection
    by combining Lasso-based screening and residual-based test statistics.
    Based on the original implementation at:
    https://github.com/moleibobliu/Distillation-CRT/
    The y-distillation is based on a given estimator and the x-distillation is
    based on a Lasso estimator.

    Parameters
    ----------
    estimator : sklearn estimator
        The base estimator used for y-distillation and prediction
        (e.g., Lasso, RandomForest, ...).
    method : str, default="predict"
        Method of the estimator to use for predictions ("predict", "predict_proba", etc.).
    estimated_coef : array-like of shape (n_features,) or None, default=None
        Pre-computed feature coefficients. If None, coefficients are estimated via Lasso.
    sigma_X : array-like of shape (n_features, n_features) or None, default=None
        Covariance matrix of X. If None, Lasso is used for X distillation.
    params_lasso_screening : dict
        Parameters for variable screening Lasso:
        - alpha : float or None - L1 regularization strength. If None, determined by CV.
        - n_alphas : int - Number of alphas for cross-validation (default: 10).
        - alphas : array-like or None - List of alpha values to try in CV (default: None).
        - alpha_max_fraction : float - Scale factor for alpha_max (default: 0.5).
        - cv : int - Cross-validation folds (default: 5).
        - tol : float - Convergence tolerance (default: 1e-6).
        - max_iter : int - Maximum iterations (default: 1000).
        - fit_intercept : bool - Whether to fit intercept (default: False).
        - selection : {'cyclic'} - Feature selection method (default: 'cyclic').
    params_lasso_distillation_x : dict or None, default=None
        Parameters for X distillation Lasso. If None, uses params_lasso_screening.
    refit : bool, default=False
        Whether to refit the model on selected features after screening.
    screening : bool, default=True
        Whether to perform variable screening step based on Lasso coefficients.
    screening_threshold : float, default=10
        Percentile threshold for screening (0-100).
        Larger values include more variables at screening.
        (screening_threshold=100 keeps all variables).
    centered : bool, default=True
        Whether to center and scale features using StandardScaler.
    n_jobs : int, default=1
        Number of parallel jobs.
    joblib_verbose : int, default=0
        Verbosity level for parallel jobs.
    fit_y : bool, default=False
        Whether to fit y using selected features instead of using estimated_coef.
    scaled_statistics : bool, default=False
        Whether to use scaled statistics when computing importance.

    Attributes
    ----------
    coefficient_ : ndarray of shape (n_features,)
        Estimated feature coefficients after screening/refitting during fit method.
    clf_x_ : list of estimators of length n_features
        Fitted models for X distillation (Lasso or None if using sigma_X).
    clf_y_ : list of estimators of length n_features
        Fitted models for y distillation
        (sklearn estimator or None if using estimated_coef and Lasso estimator).
    clf_screening_ : LassoCV or Lasso
        Fitted screening model if estimated_coef is None.
    non_selection_ : ndarray
        Indices of features not selected after screening.
    pvalues_ : ndarray of shape (n_features,)
        Computed p-values for each feature.
    importances_ : ndarray of shape (n_features,)
        Importance scores for each feature.
        Test statistics following standard normal distribution.

    Notes
    -----
    The implementation follows Liu et al. (2022), introducing distillation to
    speed up conditional randomization testing. Key steps:
    1. Optional screening using Lasso coefficients to reduce dimensionality.
    2. Distillation to estimate conditional distributions.
    3. Test statistic computation using residual correlations.
    4. P-value calculation assuming Gaussian null distribution.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        estimator,
        method: str = "predict",
        estimated_coef=None,
        sigma_X=None,
        params_lasso_screening={
            "alpha": None,
            "n_alphas": 10,
            "alphas": None,
            "alpha_max_fraction": 0.5,
            "cv": 5,
            "tol": 1e-6,
            "max_iter": 1000,
            "fit_intercept": False,
            "selection": "cyclic",
        },
        params_lasso_distillation_x=None,
        refit=False,
        screening=True,
        screening_threshold=10,
        centered=True,
        n_jobs=1,
        joblib_verbose=0,
        fit_y=False,
        scaled_statistics=False,
    ):
        self.estimator = estimator
        _check_vim_predict_method(method)
        self.estimated_coef = estimated_coef
        self.method = method
        self.sigma_X = sigma_X
        self.params_lasso_screening = params_lasso_screening
        self.params_lasso_distillation_x = params_lasso_distillation_x
        self.refit = refit
        self.screening = screening
        self.screening_threshold = screening_threshold
        self.centered = centered
        self.n_jobs = n_jobs
        self.joblib_verbose = joblib_verbose
        self.fit_y = fit_y
        self.scaled_statistics = scaled_statistics

    def fit(self, X, y):
        """
        Fit the dCRT model.

        This method fits the Distilled Conditional Randomization Test (DCRT) model
        as described in :footcite:t:`liu2022fast`. It performs optional feature
        screening using Lasso, computes coefficients, and prepares the model for
        importance and p-value computation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data matrix.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the fitted instance.

        Notes
        -----
        Main steps:
        1. Optional data centering with StandardScaler
        2. Lasso screening of variables (if no estimated coefficients provided)
        3. Feature selection based on coefficient magnitudes
        4. Model refitting on selected features (if refit=True)
        5. Fit model for future distillation

        The screening threshold controls which features are kept based on their
        Lasso coefficients. Features with coefficients below the threshold are
        set to zero.

        References
        ----------
        .. footbibliography::
        """
        if self.centered:
            X_ = StandardScaler().fit_transform(X)
        else:
            X_ = X
        y_ = y  # avoid modifying the original y
        _, n_features = X_.shape

        # screening process
        if isinstance(self.estimator, Lasso) or isinstance(self.estimator, LassoCV):
            ## Screening of variables for accelarate dCRT
            if self.estimated_coef is None:
                # base on the Theorem 2 of `liu2022fast`, the rule of screening
                # is based on a cross-validated lasso
                self.clf_screening_ = _fit_lasso(
                    X_,
                    y_,
                    n_jobs=self.n_jobs,
                    **self.params_lasso_screening,
                )
                if hasattr(self.clf_screening_, "alpha_"):
                    alpha_screening = self.clf_screening_.alpha_
                else:
                    alpha_screening = self.clf_screening_.alpha
                # update the alpha value from the estimator on all values
                self.params_lasso_screening["alpha"] = alpha_screening
                if self.params_lasso_distillation_x is not None:
                    self.params_lasso_distillation_x["alpha"] = alpha_screening
                self.coefficient_ = np.ravel(self.clf_screening_.coef_)
            else:
                warnings.warn(
                    "Precomputed coefficients were provided, so cross-validation screening will be skipped."
                )
                self.coefficient_ = self.estimated_coef
                self.screening_threshold = 100  # remove the screening process
            # noisy estimated coefficients is set to 0.0
            self.non_selection_ = np.abs(self.coefficient_) <= np.percentile(
                np.abs(self.coefficient_), 100 - self.screening_threshold
            )
            # optimisation to reduce the number of elements different to zeros
            self.coefficient_[self.non_selection_] = 0.0
            # select the variables for the screening
            if self.screening:
                if np.sum(self.non_selection_) == n_features:
                    self.clf_x_ = np.array([])
                    self.clf_y_ = np.array([])
                    return self
            else:
                self.non_selection_ = np.zeros_like(np.arange(n_features), dtype=bool)
            selection_set = np.logical_not(self.non_selection_)
            # Refit the model with the estimated support set
            if (
                self.refit
                and self.estimated_coef is None
                and np.sum(selection_set) < n_features
            ):
                self.clf_refit_ = clone(self.clf_screening_)
                self.clf_refit_.fit(X_[:, selection_set], y_)
                self.coefficient_[selection_set] = np.ravel(self.clf_refit_.coef_)
        else:
            self.coefficient_ = None
            self.non_selection_ = np.zeros_like(np.arange(n_features), dtype=bool)
            selection_set = np.logical_not(self.non_selection_)

        ## fit models
        results = Parallel(self.n_jobs, verbose=self.joblib_verbose)(
            delayed(_joblib_fit)(
                idx,
                X_,
                y_,
                self.estimator,
                sigma_X=self.sigma_X is None,
                fit_y=self.fit_y,
                params_lasso_distillation_x=(
                    self.params_lasso_distillation_x
                    if self.params_lasso_distillation_x is not None
                    else self.params_lasso_screening
                ),
            )
            for idx in np.where(selection_set)[0]
        )
        self.clf_x_ = [result[0] for result in results]
        self.clf_y_ = [result[1] for result in results]
        if self.fit_y:
            self.coefficient_ = np.array([result[2] for result in results])

        return self

    def _check_fit(self):
        """
        Check if the model has been fit before performing analysis.

        This private method verifies that all necessary attributes have been set
        during the fitting process.
        These attributes include:
        - clf_x_
        - clf_y_
        - coefficient_
        - non_selection_

        Raises
        ------
        ValueError
            If any of the required attributes are missing, indicating the model
            hasn't been fit.
        """
        if (
            not hasattr(self, "clf_x_")
            or not hasattr(self, "clf_y_")
            or not hasattr(self, "coefficient_")
            or not hasattr(self, "non_selection_")
        ):
            raise ValueError("The D0CRT requires to be fit before any analysis")

    def importance(
        self,
        X,
        y,
    ):
        """
        Compute feature importance scores using distilled CRT.

        Calculates test statistics and p-values for each feature using residual
        correlations after the distillation process.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data matrix.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        importances_ : ndarray of shape (n_features,)
            Test statistics/importance scores for each feature. For unselected features,
            the score is set to 0.

        Attributes
        ----------
        importances_ : same as return value
        pvalues_ : ndarray of shape (n_features,)
            Two-sided p-values for each feature under Gaussian null.

        Notes
        -----
        For each selected feature j:
        1. Computes residuals from regressing X_j on other features
        2. Computes residuals from regressing y on other features
        3. Calculates test statistic from correlation of residuals
        4. Computes p-value assuming standard normal distribution
        """
        self._check_fit()

        y_ = y  # avoid modifying the original y

        if self.centered:
            X_ = StandardScaler().fit_transform(X)
        else:
            X_ = X
        n_samples, n_features = X_.shape
        selection_features = np.arange(n_features)[np.logical_not(self.non_selection_)]

        ## Distillation & calculate
        list_job = []
        for index, (idx, clf_y, clf_x) in enumerate(
            zip(selection_features, self.clf_y_, self.clf_x_)
        ):
            if self.coefficient_ is not None:
                if self.fit_y:
                    coefficient_minus_idx = self.coefficient_[index]
                else:
                    coefficient_minus_idx = np.delete(np.copy(self.coefficient_), idx)
            else:
                coefficient_minus_idx = None

            list_job.append(
                delayed(_joblib_distill)(
                    idx=idx,
                    X=X_,
                    y=y_,
                    clf_y=clf_y,
                    clf_x=clf_x,
                    method=self.method,
                    sigma_X=self.sigma_X,
                    coefficient_minus_idx=coefficient_minus_idx,
                )
            )

        results = Parallel(self.n_jobs, verbose=self.joblib_verbose)(list_job)

        # contatenate result
        X_residual = np.array([result[0] for result in results])
        sigma2 = np.array([result[1] for result in results])
        y_residual = np.array([result[2] for result in results])

        # By assumming X|Z following a normal law, the exact p-value can be
        # computed with the following equation (see section 2.5 in `liu2022fast`)
        # based on the residuals of y and x.
        test_statistic_selected_variables = [
            np.dot(y_residual[i], X_residual[i])
            / np.sqrt(n_samples * sigma2[i] * np.mean(y_residual[i] ** 2))
            for i in range(X_residual.shape[0])
        ]

        if self.scaled_statistics:
            test_statistic_selected_variables = (
                test_statistic_selected_variables
                - np.mean(test_statistic_selected_variables)
            ) / np.std(test_statistic_selected_variables)

        # get the results
        test_statistic = np.zeros(n_features)
        test_statistic[selection_features] = test_statistic_selected_variables

        self.importances_ = test_statistic
        self.pvalues_ = np.minimum(2 * stats.norm.sf(np.abs(test_statistic)), 1)

        return self.importances_

    def fit_importance(self, X, y, cv=None):
        """
        Fits the model to the data and computes feature importance.

        A convenience method that combines fit() and importance() into a single call.
        First fits the dCRT model to the data, then calculates importance scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data matrix.
        y : array-like of shape (n_samples,)
            Target values.
        cv : None or int, optional (default=None)
            Not used. Included for compatibility. A warning will be issued if provided.

        Returns
        -------
        importance : ndarray of shape (n_features,)
            Feature importance scores/test statistics.
            For features not selected during screening, scores are set to 0.

        Notes
        -----
        Also sets the importances\_ and pvalues\_ attributes on the instance.
        See fit() and importance() for details on the underlying computations.
        """
        if cv is not None:
            warnings.warn("cv won't be used")
        self.fit(X, y)
        return self.importance(X, y)


def _joblib_fit(
    idx,
    X,
    y,
    estimator,
    sigma_X=False,
    fit_y=False,
    params_lasso_distillation_x={
        "n_jobs": 1,
        "random_state": 44,
        "alpha": None,
        "n_alphas": 10,
        "alphas": None,
        "alpha_max_fraction": 0.5,
    },
):
    """
    Standard Lasso distillation for least squares regression.

    This function fits a Lasso model (or other estimator) to predict X[:, idx]
    from the remaining features, and optionally fits a model to predict y from
    the remaining features.
    Used as a helper for parallel feature-wise distillation in dCRT.

    Parameters
    ----------
    idx : int
        Index of the variable to be tested.
    X : array-like of shape (n_samples, n_features)
        The input data matrix.
    y : array-like of shape (n_samples,)
        The target values.
    estimator : sklearn estimator
        The estimator used for distillation and prediction.
    sigma_X : bool, default=False
        If True, use Lasso for X distillation; if False, skip X distillation.
    fit_y : bool, default=False
        Whether to fit y using Lasso when coefficient is None.
    params_lasso_distillation_x : dict, optional
        Parameters for Lasso estimation or cross-validated Lasso for X distillation.
        (for details see the description of parameter *params_lasso_screening*
        of DOCRT)

    Returns
    -------
    clf_x : estimator or None
        Fitted Lasso model for X distillation, or None if not used.
    clf_y : estimator or None
        Fitted estimator for y distillation, or None if not used.
    coefficient_minus_idx : array-like or None
        Estimated coefficients for y prediction, or None if not computed.

    References
    ----------
    .. footbibliography::
    """
    X_minus_idx = np.delete(np.copy(X), idx, 1)

    # Distill X with least square loss
    # configure Lasso and determine the alpha
    if sigma_X:
        clf_x = _fit_lasso(
            X_minus_idx,
            X[:, idx],
            n_jobs=1,
            **params_lasso_distillation_x,
        )
    else:
        clf_x = None

    if (isinstance(estimator, Lasso) or isinstance(estimator, LassoCV)) and not fit_y:
        clf_y = None
        coefficient_minus_idx = None
    else:
        clf_y = clone(estimator)
        clf_y.fit(X_minus_idx, y)
        if fit_y:
            coefficient_minus_idx = clf_y.coef_
        else:
            coefficient_minus_idx = None
    return clf_x, clf_y, coefficient_minus_idx


def _joblib_distill(
    idx, X, y, clf_y, clf_x, method, sigma_X=None, coefficient_minus_idx=None
):
    """
    Distill the values of X and y for a single feature using least squares regression.

    This function implements the distillation process for a single feature, following
    section 2.3 of :footcite:t:`liu2022fast`. It computes the residuals for X[:, idx]
    and y after regressing out the effect of the other features, using either Lasso-based
    or covariance-based regression.

    Parameters
    ----------
    idx : int
        Index of the variable to be tested.
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target values.
    clf_y : sklearn compatible estimator
        The estimator to use for the prediction of y.
    clf_x : sklearn compatible estimator
        The estimator to use for the prediction of X minus one features.
    method : str, default="predict"
        The method to use for the prediction. Supported methods are "predict", "predict_proba" or
        "decision_function".
    sigma_X : array-like of shape (n_features, n_features) or None, default=None
        Covariance matrix of X. If provided, covariance-based regression is used for X.
    coefficient_minus_idx : ndarray of shape (n_features,) or None
        Estimated feature coefficients for y prediction, or None if not computed.

    Returns
    -------
    X_residual : ndarray of shape (n_samples,)
        Residuals after X distillation.
    sigma2 : float
        Estimated residual variance for X[:, idx].
    y_residual : ndarray of shape (n_samples,)
        Residuals after y distillation.

    References
    ----------
    .. footbibliography::
    """
    X_minus_idx = np.delete(np.copy(X), idx, 1)

    # Distill X with least square loss
    if sigma_X is None:
        n_samples = X.shape[0]
        if hasattr(clf_x, "alpha_"):
            alpha = clf_x.alpha_
        else:
            alpha = clf_x.alpha
        # get the residuals
        X_residual = X[:, idx] - clf_x.predict(X_minus_idx)
        # compute the variance of the residuals
        # In the original paper and implementation, the term:
        #  alpha * np.linalg.norm(clf.coef_, ord=1)
        # is not present and has been added without any reference actually
        sigma2 = np.linalg.norm(X_residual) ** 2 / n_samples + alpha * np.linalg.norm(
            clf_x.coef_, ord=1
        )
    else:
        # Distill X with sigma_X
        sigma_temp = np.delete(np.copy(sigma_X), idx, 0)
        b = sigma_temp[:, idx]
        A = np.delete(np.copy(sigma_temp), idx, 1)
        coefs_X = np.linalg.solve(A, b)
        X_residual = X[:, idx] - np.dot(X_minus_idx, coefs_X)
        sigma2 = sigma_X[idx, idx] - np.dot(
            np.delete(np.copy(sigma_X[idx, :]), idx), coefs_X
        )

    # Distill Y - calculate residual
    if coefficient_minus_idx is not None:
        # get the coefficients
        # compute the residuals
        y_residual = y - X_minus_idx.dot(coefficient_minus_idx.T)
    else:
        # get the residuals
        y_pred = getattr(clf_y, method)(X_minus_idx)
        if y_pred.ndim == 1:
            y_residual = y - y_pred
        else:
            y_residual = y - y_pred[:, 1]  # IIABDFI

    return X_residual, sigma2, y_residual


def _fit_lasso(
    X,
    y,
    n_jobs,
    alpha,
    alphas,
    n_alphas,
    alpha_max_fraction,
    **params,
):
    """
    Fit a LASSO regression model with optional cross-validation for alpha selection.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Target values.
    n_jobs : int
        Number of CPUs to use for cross-validation.
    alpha : float or None
        Regularization strength. If None and alphas/n_alphas not provided,
        alpha is set to alpha_max_fraction * alpha_max.
    alphas : array-like or None
        List of alphas to use for model fitting. If None and n_alphas > 0,
        alphas are set automatically.
    n_alphas : int
        Number of alphas along the regularization path. Ignored if alphas is provided.
    alpha_max_fraction : float
        Fraction of alpha_max to use when alpha, alphas, and n_alphas are not provided.
    **params : dict
        Additional parameters for Lasso or LassoCV.

    Returns
    -------
    clf : estimator
        Fitted Lasso or LassoCV model.

    Notes
    -----
    Uses cross-validation to select the best alpha if alphas or n_alphas are provided.
    Otherwise, uses a single alpha value either provided or computed from alpha_max_fraction.
    """
    if alphas is not None or n_alphas > 0:
        clf = LassoCV(
            n_alphas=n_alphas,
            alphas=alphas,
            n_jobs=n_jobs,
            **params,
        )
        clf.fit(X, y)
    else:
        if alpha is None:
            alpha = alpha_max_fraction * _alpha_max(X, y)
        clf = Lasso(
            alpha=alpha,
            **params,
        )
        clf.fit(X, y)
    return clf


def d0crt(
    estimator,
    X,
    y,
    cv=None,
    method="predict",
    estimated_coef=None,
    sigma_X=None,
    params_lasso_screening={
        "alpha": None,
        "n_alphas": 10,
        "alphas": None,
        "alpha_max_fraction": 0.5,
        "cv": 5,
        "tol": 1e-6,
        "max_iter": 1000,
        "fit_intercept": False,
        "selection": "cyclic",
    },
    params_lasso_distillation_x=None,
    refit=False,
    screening=True,
    screening_threshold=1e-1,
    centered=True,
    n_jobs=1,
    joblib_verbose=0,
    fit_y=False,
    random_state=2022,
    scaled_statistics=False,
    k_best=None,
    percentile=None,
    threshold=None,
    threshold_pvalue=None,
):
    methods = D0CRT(
        estimator=estimator,
        method=method,
        estimated_coef=estimated_coef,
        sigma_X=sigma_X,
        params_lasso_screening=params_lasso_screening,
        params_lasso_distillation_x=params_lasso_distillation_x,
        refit=refit,
        screening=screening,
        screening_threshold=screening_threshold,
        centered=centered,
        n_jobs=n_jobs,
        joblib_verbose=joblib_verbose,
        fit_y=fit_y,
        scaled_statistics=scaled_statistics,
    )
    methods.fit_importance(X, y, cv=cv)
    selection = methods.selection(
        k_best=k_best,
        percentile=percentile,
        threshold=threshold,
        threshold_pvalue=threshold_pvalue,
    )
    return selection, methods.importances_, methods.pvalues_


# use the docstring of the class for the function
d0crt.__doc__ = _aggregate_docstring(
    [
        D0CRT.__doc__,
        D0CRT.__init__.__doc__,
        D0CRT.fit_importance.__doc__,
        D0CRT.selection.__doc__,
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
