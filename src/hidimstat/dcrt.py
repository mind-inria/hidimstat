import numpy as np
import warnings
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler

from hidimstat._utils.regression import _alpha_max
from hidimstat._utils.docstring import _aggregate_docstring
from hidimstat.base_variable_importance import BaseVariableImportance


class D0CRT(BaseVariableImportance):
    """
    Implements distilled conditional randomization test (dCRT) without interactions.

    A faster version of the Conditional Randomization Test :footcite:t:`candes2018panning`
    using the distillation process from :footcite:t:`liu2022fast`. Based on original
    implementation at: https://github.com/moleibobliu/Distillation-CRT/

    Parameters
    ----------
    estimated_coef : array-like of shape (n_features,) or None, default=None
        Pre-computed feature coefficients
    sigma_X : array-like of shape (n_features, n_features) or None, default=None
        Covariance matrix of X
    params_lasso_screening : dict
        Parameters for variable screening Lasso:
        - alpha : float or None - L1 regularization strength. If None, determined by CV
        - n_alphas : int - Number of alphas for cross-validation (default: 10)
        - alphas : array-like or None - List of alpha values to try in CV (default: None)
        - alpha_max_fraction : float - Scale factor for alpha_max (default: 0.5)
        - cv : int - Cross-validation folds (default: 5)
        - tol : float - Convergence tolerance (default: 1e-6)
        - max_iter : int - Maximum iterations (default: 1000)
        - fit_intercept : bool - Whether to fit intercept (default: False)
        - selection : {'cyclic'} - Feature selection method (default: 'cyclic')
    params_lasso_distillation_x : dict or None, default=None
        Parameters for X distillation Lasso. If None, uses params_lasso_screening
    params_lasso_distillation_y : dict or None, default=None
        Parameters for y distillation Lasso. If None, uses params_lasso_screening
    refit : bool, default=False
        Whether to refit model on selected features
    screening : bool, default=True
        Whether to perform variable screening step based on Lasso coefficients
    screening_threshold : float, default=10
        Percentile threshold for screening (0-100), larger values lead to
        the inclusion of more variables at the screening stage
        (screening_threshold=100 keeps all variables).
    statistic : {'residual', 'random_forest'}, default='residual'
        Method for computing test statistics:
        - 'residual': Uses Lasso regression residuals (faster, better for linear relationships)
        - 'random_forest': Uses Random Forest predictions (captures non-linear relationships)
    centered : bool, default=True
        Whether to center and scale features using StandardScaler
    n_jobs : int, default=1
        Number of parallel jobs
    joblib_verbose : int, default=0
        Verbosity level for parallel jobs
    fit_y : bool, default=False
        Whether to fit y using selected features instead of using estimated_coef
    n_tree : int, default=100
        Number of trees for random forest when statistic='random_forest'
    problem_type : {'regression', 'classification'}, default='regression'
        Type of prediction problem when using random forest
    scaled_statistics : bool, optional (default=False)
        Whether to use scaled statistics when computing importance.
    random_state : int, default=2022
        Random seed for reproducibility

    Attributes
    ----------
    coefficient : ndarray of shape (n_features,)
        Estimated feature coefficients
    clf_x_residual : list of estimators of length n_features
        Fitted models for X distillation (Lasso or None if using sigma_X)
    clf_y_residual : list of estimators of length n_features
        Fitted models for y distillation (Lasso/RandomForest or None if using estimated_coef)
    clf_screening : LassoCV or Lasso
        Fitted screening model if estimated_coef=None
    selection_features : ndarray of shape (n_features,)
        Boolean mask indicating selected features after screening
    sigma2 : ndarray of shape (n_selected_features,)
        Estimated residual variances for selected features
    ts : ndarray of shape (n_features,)
        Test statistics following standard normal distribution
    X_residual : ndarray of shape (n_selected_features, n_samples)
        Residuals from X distillation for selected features
    y_residual : ndarray of shape (n_selected_features, n_samples)
        Residuals from y distillation for selected features

    Notes
    -----
    The implementation follows Liu et al. (2022) which introduces distillation to
    speed up conditional randomization testing. Key steps:
    1. Optional screening using Lasso coefficients to reduce dimensionality
    2. Distillation to estimate conditional distributions
    3. Test statistic computation using residual correlations or random forests
    4. P-value calculation assuming Gaussian null distribution

    See Also
    --------
    sklearn.linear_model.Lasso : Basic Lasso regression
    sklearn.linear_model.LassoCV : Lasso with cross-validation
    sklearn.ensemble.RandomForestRegressor : Random forest for regression
    sklearn.ensemble.RandomForestClassifier : Random forest for classification
    sklearn.preprocessing.StandardScaler : Feature standardization

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
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
        params_lasso_distillation_y=None,
        refit=False,
        screening=True,
        screening_threshold=10,
        statistic="residual",
        centered=True,
        n_jobs=1,
        joblib_verbose=0,
        fit_y=False,
        n_tree=100,
        problem_type="regression",
        scaled_statistics=False,
        random_state=2022,
    ):
        self.estimated_coef = estimated_coef
        self.sigma_X = sigma_X
        self.params_lasso_screening = params_lasso_screening
        self.params_lasso_distillation_x = params_lasso_distillation_x
        self.params_lasso_distillation_y = params_lasso_distillation_y
        self.refit = refit
        self.screening = screening
        self.screening_threshold = screening_threshold
        self.statistic = statistic
        self.centered = centered
        self.n_jobs = n_jobs
        self.joblib_verbose = joblib_verbose
        self.fit_y = fit_y
        self.n_tree = n_tree
        self.problem_type = problem_type
        self.scaled_statistics = scaled_statistics
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the dCRT model.
        Based on the paper by :footcite:t:`liu2022fast` for fast conditional
        randomization testing.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data matrix where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.

        Notes
        -----
        The method follows these main steps:
        1. Optional data centering using StandardScaler
        2. Variable screening using LASSO (if not provided with estimated coefficients)
        3. Feature selection based on coefficient magnitudes
        4. Model refitting on selected features (if refit=True)
        5. Distillation process using either residual or random forest statistics

        The screening threshold determines which features are considered significant
        based on their LASSO coefficients. Features with coefficients below the
        threshold percentile are set to zero.

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

        ## Distillation & calculate
        if self.statistic == "residual":
            ## Screening of variables for accelarate dCRT
            if self.estimated_coef is None:
                # base on the Theorem 2 of `liu2022fast`, the rule of screening
                # is based on a cross-validated lasso
                self.clf_screening, alpha_screening = _fit_lasso(
                    X_,
                    y_,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    **self.params_lasso_screening,
                )
                # update the alpha value from the estiamtor on all values
                self.params_lasso_screening["alpha"] = alpha_screening
                if self.params_lasso_distillation_x is not None:
                    self.params_lasso_distillation_x["alpha"] = alpha_screening
                if self.params_lasso_distillation_y is not None:
                    self.params_lasso_distillation_y["alpha"] = alpha_screening
                self.coefficient_ = np.ravel(self.clf_screening.coef_)
            else:
                self.coefficient_ = self.estimated_coef
                self.screening_threshold = 100  # remove the screening process
            # noisy estimated coefficients is set to 0.0
            self.non_selection = np.where(
                np.abs(self.coefficient_)
                <= np.percentile(
                    np.abs(self.coefficient_), 100 - self.screening_threshold
                )
            )[0]
            # optimisation to reduce the number of elements different to zeros
            self.coefficient_[self.non_selection] = 0.0
            # select the variables for the screening
            if self.screening:
                selection_set = np.setdiff1d(np.arange(n_features), self.non_selection)
                if selection_set.size == 0:
                    self.selection_features = np.array([])
                    self.X_residual = np.array([])
                    self.sigma2 = np.array([])
                    self.y_residual = np.array([])
                    self.clf_x_residual = np.array([])
                    self.clf_y_residual = np.array([])
                    return self
            else:
                self.non_selection = []
                selection_set = np.arange(n_features)
            # Refit the model with the estimated support set
            if (
                self.refit
                and self.estimated_coef is None
                and selection_set.size < n_features
            ):
                self.clf_refit = clone(self.clf_screening)
                self.clf_refit.fit(X_[:, selection_set], y_)
                self.coefficient_[selection_set] = np.ravel(self.clf_refit.coef_)
            # For distillation of X use least_square loss
            results = Parallel(self.n_jobs, verbose=self.joblib_verbose)(
                delayed(_lasso_distillation_residual)(
                    X_,
                    y_,
                    idx,
                    coefficient=self.coefficient_,
                    sigma_X=self.sigma_X,
                    params_lasso_distillation_x=(
                        self.params_lasso_distillation_x
                        if self.params_lasso_distillation_x is not None
                        else self.params_lasso_screening
                    ),
                    params_lasso_distillation_y=(
                        self.params_lasso_distillation_y
                        if self.params_lasso_distillation_y is not None
                        else self.params_lasso_screening
                    ),
                    fit_y=self.fit_y,
                    n_jobs=1,  # the function is already called in parallel
                    random_state=self.random_state,
                )
                for idx in selection_set
            )
        elif self.statistic == "random_forest":
            selection_set = range(n_features)
            self.non_selection = []
            # For distillation of X use least_square loss
            results = Parallel(self.n_jobs, verbose=self.joblib_verbose)(
                delayed(_rf_distillation)(
                    X_,
                    y_,
                    idx,
                    sigma_X=self.sigma_X,
                    n_tree=self.n_tree,
                    problem_type=self.problem_type,
                    n_jobs=1,  # the function is already called in parallel
                    random_state=self.random_state,
                    params_lasso_distillation_x=(
                        self.params_lasso_distillation_x
                        if self.params_lasso_distillation_y is not None
                        else self.params_lasso_screening
                    ),
                )
                for idx in selection_set
            )
        else:
            raise ValueError(f"{self.statistic} statistic is not supported.")
        # contatenate result
        self.selection_features = np.ones((n_features,), dtype=bool)
        self.selection_features[self.non_selection] = 0
        self.X_residual = np.array([result[0] for result in results])
        self.sigma2 = np.array([result[1] for result in results])
        self.y_residual = np.array([result[2] for result in results])
        self.clf_x_residual = np.array([result[3] for result in results])
        self.clf_y_residual = np.array([result[4] for result in results])
        return self

    def _check_fit(self):
        """
        Check if the model has been fit before performing analysis.

        This private method verifies that all necessary attributes have been set
        during the fitting process.
        These attributes include:
        - selection_features
        - X_residual
        - sigma2
        - y_residual
        - clf_x_residual
        - clf_y_residual

        Raises
        ------
        ValueError
            If any of the required attributes are missing, indicating the model
            hasn't been fit.
        """
        if (
            not hasattr(self, "selection_features")
            or not hasattr(self, "X_residual")
            or not hasattr(self, "sigma2")
            or not hasattr(self, "y_residual")
            or not hasattr(self, "clf_x_residual")
            or not hasattr(self, "clf_y_residual")
        ):
            raise ValueError("The D0CRT requires to be fit before any analysis")

    def importance(
        self,
        X=None,
        y=None,
    ):
        """
        Calculate p-values and identify significant features using the dCRT test
        statistics. This function processes the results from dCRT to identify
        statistically significant features while controlling for false discoveries.
        It assumes test statistics follow a Gaussian distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            (not used) Testing data matrix where n_samples is the number of samples
            and n_features is the number of features.
            A warning will be issued if provided.
        y : array-like of shape (n_samples,), default=None
            (not used) Target values of testing dataset.
            A warning will be issued if provided.

        Returns
        -------
        importances_ : ndarray of shape (n_features,)
            Importance scores for all features

        Notes
        -----
        The function computes test statistics as correlations between residuals,
        optionally scales them, and converts to p-values using a Gaussian null.
        """
        if X is not None:
            warnings.warn("X won't be used")
        if y is not None:
            warnings.warn("y won't be used")

        self._check_fit()
        n_features = self.selection_features.shape[0]
        n_samples = self.X_residual.shape[1]

        # By assumming X|Z following a normal law, the exact p-value can be
        # computed with the following equation (see section 2.5 in `liu2022fast`)
        # based on the residuals of y and x.
        ts_selected_variables = [
            np.dot(self.y_residual[i], self.X_residual[i])
            / np.sqrt(n_samples * self.sigma2[i] * np.mean(self.y_residual[i] ** 2))
            for i in range(self.X_residual.shape[0])
        ]

        if self.scaled_statistics:
            ts_selected_variables = (
                ts_selected_variables - np.mean(ts_selected_variables)
            ) / np.std(ts_selected_variables)

        # get the results
        self.ts = np.zeros(n_features)
        self.ts[self.selection_features] = ts_selected_variables

        self.pvalues_ = np.minimum(2 * stats.norm.sf(np.abs(self.ts)), 1)
        self.importances_ = self.pvalues_

        return self.importances_

    def fit_importance(self, X, y, cv=None):
        """
        Fits the model to the data and computes feature importance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The target values.
        cv : None or int, optional (default=None)
            (not used) Cross-validation parameter.
            A warning will be issued if provided.

        Returns
        -------
        importance : array-like
            The computed feature importance scores.
        """
        if cv is not None:
            warnings.warn("cv won't be used")
        self.fit(X, y)
        return self.importance()


def _x_distillation_lasso(
    X,
    idx,
    sigma_X=None,
    n_jobs=1,
    random_state=0,
    params_lasso_distillation_x=None,
):
    """
    Distill variable X[:, idx] using Lasso regression on remaining variables.

    This function implements the distillation process to estimate the conditional
    distribution of X[:, idx] given the remaining variables, using either Lasso
    regression or a known covariance matrix.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples.
    idx : int
        Index of the variable to be distilled.
    sigma_X : {array-like, sparse matrix} of shape (n_features, n_features), default=None
        The covariance matrix of X. If provided, used instead of Lasso regression.
    n_jobs : int, default=1
        Number of CPUs to use for cross-validation.
    random_state : int, default=0
        Random seed for reproducibility.
    params_lasso_distillation_x : dict or None, default=None
        Parameters for main Lasso estimation or crossvalidation Lasso, including:
        - alpha : float - L1 regularization strength. If None, determined by CV.
        - n_alphas : int, default=0 - Number of alphas for cross-validation.
        - alphas : array-like, default=None - List of alpha values to try in CV.
        - alpha_max_fraction : float, default=0.5 - Scale factor for alpha_max.

    Returns
    -------
    X_res : ndarray of shape (n_samples,)
        The residuals after distillation.
    sigma2 : float
        The estimated variance of the residuals.
    clf : estimator or None
        The fitted Lasso model, or None if sigma_X was used.
    """
    n_samples = X.shape[0]
    X_minus_idx = np.delete(np.copy(X), idx, 1)

    if sigma_X is None:
        # Distill X with least square loss
        # configure Lasso and determine the alpha
        clf, alpha = _fit_lasso(
            X_minus_idx,
            X[:, idx],
            n_jobs=n_jobs,
            random_state=random_state,
            **params_lasso_distillation_x,
        )

        # get the residuals
        X_res = X[:, idx] - clf.predict(X_minus_idx)
        # compute the variance of the residuals
        # In the original paper and implementation, the term:
        #  alpha * np.linalg.norm(clf.coef_, ord=1)
        # is not present and has been added without any reference actually
        sigma2 = np.linalg.norm(X_res) ** 2 / n_samples + alpha * np.linalg.norm(
            clf.coef_, ord=1
        )

    else:
        clf = None
        # Distill X with sigma_X
        sigma_temp = np.delete(np.copy(sigma_X), idx, 0)
        b = sigma_temp[:, idx]
        A = np.delete(np.copy(sigma_temp), idx, 1)
        coefs_X = np.linalg.solve(A, b)
        X_res = X[:, idx] - np.dot(X_minus_idx, coefs_X)
        sigma2 = sigma_X[idx, idx] - np.dot(
            np.delete(np.copy(sigma_X[idx, :]), idx), coefs_X
        )
    return X_res, sigma2, clf


def _lasso_distillation_residual(
    X,
    y,
    idx,
    coefficient=None,
    sigma_X=None,
    n_jobs=1,
    fit_y=False,
    random_state=42,
    params_lasso_distillation_x={
        "alpha": None,
        "n_alphas": 10,
        "alphas": None,
        "alpha_max_fraction": 0.5,
    },
    params_lasso_distillation_y={
        "alpha": None,
        "n_alphas": 10,
        "alphas": None,
        "alpha_max_fraction": 0.5,
    },
):
    """
    Standard Lasso Distillation for least squares regression.

    This function implements the distillation process following :footcite:t:`liu2022fast`
    section 2.3. It distills both X[:, idx] and y to compute test statistics.
    It's based on least square loss regression.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target values.
    idx : int
        Index of the variable to be tested.
    coefficient : array-like of shape (n_features,), default=None
        Pre-computed coefficients for y prediction. If None, computed via Lasso.
    sigma_X : {array-like, sparse matrix} of shape (n_features, n_features), default=None
        The covariance matrix of X.
    n_jobs : int, default=1
        Number of CPUs to use.
    fit_y : bool, default=False
        Whether to fit y using Lasso when coefficient is None.
    random_state : int, default=42
        Random seed for reproducibility.
    params_lasso_distillation_x : dict
        Parameters for main Lasso estimation or crossvalidation Lasso, including:
        - alpha : float - L1 regularization strength. If None, determined by CV.
        - n_alphas : int, default=0 - Number of alphas for cross-validation.
        - alphas : array-like, default=None - List of alpha values to try in CV.
        - alpha_max_fraction : float, default=0.5 - Scale factor for alpha_max.
    params_lasso_distillation_y : dict
        Same as params_lasso_distillation_x.

    Returns
    -------
    X_residual : ndarray of shape (n_samples,)
        The residuals after X distillation.
    sigma2 : float
        The estimated variance of the residuals.
    y_residual : ndarray of shape (n_samples,)
        The residuals after y distillation.
    clf_x_residual : Lasso or None
        The fitted Lasso model for X distillation, or None if sigma_X was used.
    clf_y : Lasso or None
        The fitted Lasso model for y distillation, or None if coefficient was provided.

    References
    ----------
    .. footbibliography::
    """
    X_minus_idx = np.delete(np.copy(X), idx, 1)

    # Distill X with least square loss
    X_residual, sigma2, clf_x_residual = _x_distillation_lasso(
        X,
        idx,
        sigma_X,
        n_jobs=n_jobs,
        random_state=random_state + 2,  # avoid the same seed as the main function
        params_lasso_distillation_x=params_lasso_distillation_x,
    )

    # Distill Y - calculate residual
    # get the coefficients
    if fit_y:
        clf_y, alpha_null = _fit_lasso(
            X_minus_idx,
            y,
            n_jobs=n_jobs,
            random_state=random_state,
            **params_lasso_distillation_y,
        )
        coefficient_minus_idx = clf_y.coef_
    elif coefficient is not None:
        coefficient_minus_idx = np.delete(np.copy(coefficient), idx)
        clf_y = None
    else:
        raise ValueError("Either fit_y is true or coefficient must be provided.")

    # compute the residuals
    y_residual = y - X_minus_idx.dot(coefficient_minus_idx)

    return X_residual, sigma2, y_residual, clf_x_residual, clf_y


def _rf_distillation(
    X,
    y,
    idx,
    sigma_X=None,
    n_jobs=1,
    problem_type="regression",
    n_tree=100,
    random_state=42,
    params_lasso_distillation_x={
        "alpha": None,
        "n_alphas": 10,
        "alphas": None,
        "alpha_max_fraction": 0.5,
    },
):
    """
    Random Forest based distillation for both regression and classification.

    This function implements the distillation process using Random Forest for y
    and Lasso for X[:, idx]. It supports both regression and binary classification
    problems.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target values (class labels or regression targets).
    idx : int
        Index of the variable to be tested.
    sigma_X : {array-like, sparse matrix} of shape (n_features, n_features), default=None
        The covariance matrix of X.
    n_jobs : int, default=1
        Number of CPUs to use.
    problem_type : {'regression', 'classification'}, default='regression'
        The type of prediction problem.
    n_tree : int, default=100
        Number of trees in the Random Forest.
    random_state : int, default=42
        Random seed for reproducibility.
    params_lasso_distillation_x : dict
        Parameters for main Lasso estimation or crossvalidation Lasso, including:
        - alpha : float - L1 regularization strength. If None, determined by CV.
        - n_alphas : int, default=0 - Number of alphas for cross-validation.
        - alphas : array-like, default=None - List of alpha values to try in CV.
        - alpha_max_fraction : float, default=0.5 - Scale factor for alpha_max.

    Returns
    -------
    X_residual : ndarray of shape (n_samples,)
        The residuals after X distillation.
    sigma2 : float
        The estimated variance of the residuals.
    y_residual : ndarray of shape (n_samples,)
        The residuals after y distillation.
    clf_x_residual : Lasso or None
        The fitted Lasso model for X distillation, or None if sigma_X was used.
    clf_y : RandomForestRegressor or RandomForestClassifier
        The fitted Random Forest model for y distillation.
    Notes
    -----
    For classification, the function uses probability predictions from
    RandomForestClassifier and assumes binary classification (uses class 1
    probability only).
    """
    X_minus_idx = np.delete(np.copy(X), idx, 1)

    # Distill X with least square loss
    X_residual, sigma2, clf_x_residual = _x_distillation_lasso(
        X,
        idx,
        sigma_X,
        n_jobs=n_jobs,
        random_state=random_state + 2,  # avoid the same seed as the main function
        params_lasso_distillation_x=params_lasso_distillation_x,
    )

    # Distill Y
    # get the residuals
    if problem_type == "regression":
        clf_y = RandomForestRegressor(
            n_estimators=n_tree, random_state=random_state, n_jobs=n_jobs
        )
        clf_y.fit(X_minus_idx, y)
        y_residual = y - clf_y.predict(X_minus_idx)
    elif problem_type == "classification":
        clf_y = RandomForestClassifier(
            n_estimators=n_tree, random_state=random_state, n_jobs=n_jobs
        )
        clf_y.fit(X_minus_idx, y)
        y_residual = y - clf_y.predict_proba(X_minus_idx)[:, 1]  # IIABDFI
    return (X_residual, sigma2, y_residual, clf_x_residual, clf_y)


def _fit_lasso(
    X,
    y,
    n_jobs,
    alpha,
    alphas,
    n_alphas,
    alpha_max_fraction,
    random_state,
    **params,
):
    """
    Fits a LASSO regression model with optional cross-validation for alpha selection.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data
    y : array-like of shape (n_samples,)
        Target values
    n_jobs : int
        Number of CPUs for cross-validation
    alpha : float
        Constant that multiplies the L1 term. If None and alphas/n_alphas not provided,
        alpha is set to alpha_max_fraction * alpha_max.
    alphas : array-like
        List of alphas where to compute the models. If None and n_alphas > 0,
        alphas are set automatically.
    n_alphas : int
        Number of alphas along the regularization path. Ignored if alphas is provided.
    alpha_max_fraction : float
        Fraction of alpha_max to use when alpha, alphas, and n_alphas are not provided.
    random_state : int, RandomState instance or None
        Random seed for reproducibility
    **params : dict
        Additional parameters for Lasso/LassoCV

    Returns
    -------
    clf : estimator
        Fitted Lasso/LassoCV model
    alpha : float
        Selected alpha value

    Notes
    -----
    Uses cross-validation to select the best alpha if alphas or n_alphas provided.
    Otherwise, uses a single alpha value either provided or computed from alpha_max_fraction.
    """
    if alphas is not None or n_alphas > 0:
        clf = LassoCV(
            n_alphas=n_alphas,
            alphas=alphas,
            n_jobs=n_jobs,
            random_state=random_state,
            **params,
        )
        clf.fit(X, y)
        alpha = clf.alpha_
    else:
        if alpha is None:
            alpha = alpha_max_fraction * _alpha_max(X, y)
        clf = Lasso(
            alpha=alpha,
            random_state=random_state,
            **params,
        )
        clf.fit(X, y)
    return clf, alpha


def d0crt(
    X,
    y,
    cv=None,
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
    params_lasso_distillation_y=None,
    refit=False,
    screening=True,
    screening_threshold=1e-1,
    statistic="residual",
    centered=True,
    n_jobs=1,
    joblib_verbose=0,
    fit_y=False,
    n_tree=100,
    problem_type="regression",
    random_state=2022,
    scaled_statistics=False,
    k_best=None,
    percentile=None,
    threshold=None,
    threshold_pvalue=None,
):
    methods = D0CRT(
        estimated_coef=estimated_coef,
        sigma_X=sigma_X,
        params_lasso_screening=params_lasso_screening,
        params_lasso_distillation_x=params_lasso_distillation_x,
        params_lasso_distillation_y=params_lasso_distillation_y,
        refit=refit,
        screening=screening,
        screening_threshold=screening_threshold,
        statistic=statistic,
        centered=centered,
        n_jobs=n_jobs,
        joblib_verbose=joblib_verbose,
        fit_y=fit_y,
        n_tree=n_tree,
        problem_type=problem_type,
        random_state=random_state,
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
    selection: binary array-like of shape (n_features)
        Binary array of the seleted features
    importance : array-like of shape (n_features)
        The computed feature importance scores.
    pvalues : array-like of shape (n_features)
        The computed significant of feature for the prediction.
    """,
)
