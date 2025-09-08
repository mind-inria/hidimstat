import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler

from hidimstat._utils.docstring import _aggregate_docstring
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
        Method of the estimator to use for predictions ("predict", "predict_proba",
        "decision_function").
    estimated_coef : array-like of shape (n_features,) or None, default=None
        Pre-computed feature coefficients. If None, coefficients are estimated via
        Lasso.
    sigma_X : array-like of shape (n_features, n_features) or None, default=None
        Covariance matrix of X. If None, Lasso is used for X distillation.
    lasso_screening : sklearn estimator
        Estimator for variable screening (typically LassoCV or Lasso).
    model_distillation_x : sklearn estimator
        Estimator for X distillation (typically LassoCV or Lasso).
    refit : bool, default=False
        Whether to refit the model on selected features after screening.
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
        Controls y-distillation behavior:
        - If False and the estimator is linear, the sub-model predicting y from X^{-j}
        is created by simply removing the idx-th coefficient from the full model
        (no fitting is performed).
        - If True, fits a clone of `estimator` on (X^{-j}, y)
        - For non-linear estimators, always fits a clone of `estimator` on (X^{-j}, y)
        regardless of fit_y.
    scaled_statistics : bool, default=False
        Whether to use scaled statistics when computing importance.
    random_state : int, default=2022
        Random seed for reproducibility.
    reuse_screening_model: bool, default=True
        Whether to reuse the screening model for y-distillation.

    Attributes
    ----------
    coefficient_ : ndarray of shape (n_features,)
        Estimated feature coefficients after screening/refitting during fit method.
    selection_set : ndarray of shape (n_features,)
        Boolean mask indicating selected features after screening.
    model_x_ : list of estimators
        Fitted models for X distillation (Lasso or None if using sigma_X).
    model_y_ : list of estimators
        Fitted models for y distillation (sklearn estimator or None if using
        estimated_coef and Lasso estimator).
    screening_coefficient_ : ndarray
        Coefficients from the screening model (LassoCV or Lasso).
    importances_ : ndarray of shape (n_features,)
        Importance scores for each feature. Test statistics following standard normal
        distribution.
    pvalues_ : ndarray of shape (n_features,)
        Computed p-values for each feature.

    Notes
    -----
    The implementation follows Liu et al. (2022), introducing distillation to
    speed up conditional randomization testing. Key steps:
    1. Optional screening using Lasso coefficients to reduce dimensionality.
    2. Distillation to estimate conditional distributions.
    3. Test statistic computation using residual correlations.
    4. P-value calculation assuming Gaussian null distribution.

    The implementation currently allows flexible models for the y-distillation step.
    However, the x-distillation step only supports linear models.

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
        lasso_screening=LassoCV(n_alphas=10, cv=5, tol=1e-6, fit_intercept=False),
        model_distillation_x=LassoCV(n_jobs=1, n_alphas=10),
        refit=False,
        screening_threshold=10,
        centered=True,
        n_jobs=1,
        joblib_verbose=0,
        fit_y=False,
        scaled_statistics=False,
        random_state=None,
        reuse_screening_model=True,
    ):
        self.estimator = estimator
        _check_vim_predict_method(method)
        self.estimated_coef = estimated_coef
        self.method = method
        self.sigma_X = sigma_X
        self.lasso_screening = lasso_screening
        self.model_distillation_x = model_distillation_x
        self.refit = refit
        self.screening_threshold = screening_threshold
        self.centered = centered
        self.n_jobs = n_jobs
        self.joblib_verbose = joblib_verbose
        self.fit_y = fit_y
        self.scaled_statistics = scaled_statistics
        self.random_state = random_state
        self.is_lasso = isinstance(estimator, (Lasso, LassoCV))
        self.reuse_screening_model = reuse_screening_model

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

        if self.screening_threshold is not None:
            if self.estimated_coef is not None:
                warnings.warn(
                    "Precomputed coefficients were provided, screening is skipped and "
                    "screening_threshold is set to 100."
                )
                self.selection_set_ = np.ones(X.shape[1], dtype=bool)
            else:
                self.selection_set, self.lasso_model_ = run_lasso_screening(
                    X_,
                    y_,
                    lasso_model=self.lasso_screening,
                    estimated_coef=self.estimated_coef,
                    screening_threshold=self.screening_threshold,
                )
                if self.refit:
                    self.estimator.fit(X_[:, self.selection_set], y_)
        else:
            self.selection_set = np.ones(X_.shape[1], dtype=bool)

        if self.estimated_coef is not None:
            self.coefficient_ = self.estimated_coef
        elif self.reuse_screening_model and (self.screening_threshold is not None):
            self.coefficient_ = self.lasso_model_.coef_
            # optimisation to reduce the number of elements different to zeros
            self.coefficient_[~self.selection_set] = 0
        else:
            self.estimator.fit(X_[:, self.selection_set], y_)
            if hasattr(self.estimator, "coef_"):
                self.coefficient_ = self.estimator.coef_
                # optimisation to reduce the number of elements different to zeros
                # TODO: the multi-dimensional indexing should be removed when the
                # classification is properly handled
                self.coefficient_[..., ~self.selection_set] = 0
            else:
                self.coefficient_ = None

        ## fit models
        results = Parallel(self.n_jobs, verbose=self.joblib_verbose)(
            delayed(_joblib_fit)(
                idx=idx,
                X=X_,
                y=y_,
                estimator=self.estimator,
                sigma_X=self.sigma_X is None,
                fit_y=self.fit_y,
                model_distillation_x=self.model_distillation_x,
            )
            for idx in np.where(self.selection_set)[0]
        )
        self.model_x_ = [result[0] for result in results]
        self.model_y_ = [result[1] for result in results]
        if self.fit_y:
            self.coefficient_ = np.array([result[2] for result in results])

        return self

    def _check_fit(self):
        """
        Check if the model has been fit before performing analysis.

        This private method verifies that all necessary attributes have been set
        during the fitting process.
        These attributes include:
        - model_x_
        - model_y_
        - coefficient_
        - selection_set

        Raises
        ------
        ValueError
            If any of the required attributes are missing, indicating the model
            hasn't been fit.
        """
        if (
            not hasattr(self, "model_x_")
            or not hasattr(self, "model_y_")
            or not hasattr(self, "coefficient_")
            or not hasattr(self, "selection_set")
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
        selection_features = np.arange(n_features)[self.selection_set]

        ## Distillation & calculate
        list_job = []
        for index, (idx, model_y, model_x) in enumerate(
            zip(selection_features, self.model_y_, self.model_x_)
        ):
            if self.is_lasso:
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
                    model_y=model_y,
                    model_x=model_x,
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
    idx, X, y, estimator, sigma_X=False, fit_y=False, model_distillation_x=None
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
        Controls y-distillation behavior:
        - If False and the estimator is linear, the sub-model predicting y from X^{-j}
        is created by simply removing the idx-th coefficient from the full model
        (no fitting is performed).
        - If True, fits a clone of `estimator` on (X^{-j}, y)
        - For non-linear estimators, always fits a clone of `estimator` on (X^{-j}, y)
        regardless of fit_y.
    model_distillation_x : sklearn estimator or None
        The model to use for distillation of X, or None if not used.

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
        model_x = clone(model_distillation_x)
        model_x.fit(X_minus_idx, X[:, idx])
    else:
        model_x = None

    if (isinstance(estimator, Lasso) or isinstance(estimator, LassoCV)) and not fit_y:
        model_y = None
        coefficient_minus_idx = None
    else:
        model_y = clone(estimator)
        model_y.fit(X_minus_idx, y)
        if fit_y:
            coefficient_minus_idx = model_y.coef_
        else:
            coefficient_minus_idx = None
    return model_x, model_y, coefficient_minus_idx


def _joblib_distill(
    idx, X, y, model_y, model_x, method, sigma_X=None, coefficient_minus_idx=None
):
    """
    Distill the values of X and y for a single feature using least squares regression.

    This function implements the distillation process for a single feature, following
    section 2.3 of :footcite:t:`liu2022fast`. It computes the residuals for X[:, idx]
    and y after regressing out the effect of the other features, using either
    Lasso-based or covariance-based regression.

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
        The method to use for the prediction. Supported methods are "predict",
        "predict_proba" or "decision_function".
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
        if hasattr(model_x, "alpha_"):
            alpha = model_x.alpha_
        else:
            alpha = model_x.alpha
        # get the residuals
        X_residual = X[:, idx] - model_x.predict(X_minus_idx)
        # compute the variance of the residuals
        # In the original paper and implementation, the term:
        #  alpha * np.linalg.norm(clf.coef_, ord=1)
        # is not present and has been added without any reference actually
        sigma2 = np.linalg.norm(X_residual) ** 2 / n_samples + alpha * np.linalg.norm(
            model_x.coef_, ord=1
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
        y_pred = getattr(model_y, method)(X_minus_idx)
        if y_pred.ndim == 1:
            y_residual = y - y_pred
        else:
            y_residual = y - y_pred[:, 1]  # IIABDFI

    return X_residual, sigma2, y_residual


def run_lasso_screening(
    X,
    y,
    lasso_model=LassoCV(fit_intercept=False),
    estimated_coef=None,
    screening_threshold=10,
):
    """
    Perform Lasso screening for feature selection.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix.
    y : array-like of shape (n_samples,)
        Target values.
    lasso_model : sklearn estimator or None, default=LassoCV(fit_intercept=False)
        Estimator for variable screening (typically LassoCV or Lasso).
    screening_threshold : float
        Percentile threshold for screening (0-100).

    Returns
    -------
    selection_set : ndarray of shape (n_features,)
        Boolean mask indicating selected features.
    lasso_model : sklearn estimator
        Fitted Lasso model used for screening.
    """
    if not (isinstance(lasso_model, LassoCV) or isinstance(lasso_model, Lasso)):
        raise ValueError("lasso_model must be an instance of Lasso or LassoCV")
    lasso_model.fit(X, y)
    selection_set = np.abs(lasso_model.coef_) >= np.percentile(
        np.abs(lasso_model.coef_), 100 - screening_threshold
    )
    return selection_set, lasso_model


def d0crt(
    estimator,
    X,
    y,
    cv=None,
    method="predict",
    estimated_coef=None,
    sigma_X=None,
    lasso_screening=LassoCV(
        n_alphas=10,
        cv=5,
        tol=1e-6,
        fit_intercept=False,
    ),
    model_distillation_x=LassoCV(
        n_jobs=1,
        n_alphas=10,
    ),
    refit=False,
    screening_threshold=10,
    centered=True,
    n_jobs=1,
    joblib_verbose=0,
    fit_y=False,
    scaled_statistics=False,
    random_state=None,
    reuse_screening_model=True,
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
        lasso_screening=lasso_screening,
        model_distillation_x=model_distillation_x,
        refit=refit,
        screening_threshold=screening_threshold,
        centered=centered,
        n_jobs=n_jobs,
        joblib_verbose=joblib_verbose,
        fit_y=fit_y,
        scaled_statistics=scaled_statistics,
        random_state=random_state,
        reuse_screening_model=reuse_screening_model,
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
