import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.covariance import LedoitWolf
from sklearn.utils.validation import check_memory
from sklearn.preprocessing import StandardScaler

from hidimstat._utils.docstring import _aggregate_docstring
from hidimstat.statistical_tools.gaussian_distribution import GaussianDistribution
from hidimstat.statistical_tools.lasso_test import lasso_statistic_with_sampling
from hidimstat.base_variable_importance import (
    BaseVariableImportance,
    _empirical_pval,
)


class ModelXKnockoff(BaseVariableImportance):
    """
    Model-X Knockoff

    This module implements the Model-X knockoff inference procedure, which is an approach
    to control the False Discovery Rate (FDR) based on :footcite:t:`candes2018panning`.
    The original implementation can be found at
    https://github.com/msesia/knockoff-filter/blob/master/R/knockoff/R/knockoff_filter.R
    The noisy variables are generated with second-order knockoff variables using the equi-correlated method.

    In addition, this function generates multiple sets of Gaussian knockoff variables and calculates
    the test statistics for each set. It then aggregates the test statistics across
    the sets to improve stability and power.

    Parameters
    ----------
    X : 2D ndarray (n_samples, n_features)
        The design matrix.

    y : 1D ndarray (n_samples, )
        The target vector.

    estimator : sklearn estimator instance or a cross validation instance
        The estimator used for fitting the data and computing the test statistics.
        This can be any estimator with a `fit` method that accepts a 2D array and
        a 1D array, and a `coef_` attribute that returns a 1D array of coefficients.
        Examples include LassoCV, LogisticRegressionCV, and LinearRegression.

        *Configuration example:*

        | LassoCV(alphas=alphas, n_jobs=None, verbose=0, max_iter=1000,
        | cv=KFold(n_splits=5, shuffle=True, random_state=0), tol=1e-8)

        | LogisticRegressionCV(penalty="l1", max_iter=1000, solver="liblinear",
        | cv=KFold(n_splits=5, shuffle=True, random_state=0), n_jobs=None,
        | tol=1e-8)

        | LogisticRegressionCV(penalty="l2", max_iter=1000, n_jobs=None,
        | verbose=0, cv=KFold(n_splits=5, shuffle=True, random_state=0),
        | tol=1e-8,)

    preconfigure_estimator : callable, default=preconfigure_estimator_LassoCV
        A function that configures the estimator for the Model-X knockoff procedure.
        If provided, this function will be called with the estimator, X, X_tilde, and y
        as arguments, and should modify the estimator in-place.

    fdr : float, default=0.1
        The desired controlled False Discovery Rate (FDR) level.

    centered : bool, default=True
        Whether to standardize the data before performing the inference procedure.

    cov_estimator : estimator object, default=LedoitWolf()
        Estimator for empirical covariance matrix.

    joblib_verbose : int, default=0
        Verbosity level for parallel jobs.

    n_bootstraps : int, default=1
        Number of bootstrap samples for aggregation.

    n_jobs : int, default=1
        Number of parallel jobs.

    random_state : int or None, default=None
        The random seed used to generate the Gaussian knockoff variables.

    tol_gauss : float, default=1e-14
        A tolerance value used for numerical stability in the calculation of
        the Cholesky decomposition in the gaussian generation function.

    memory : str or Memory object, default=None
        Used to cache the output of the clustering and inference computation.
        By default, no caching is done. If provided, it should be the path
        to the caching directory or a joblib.Memory object.

    Returns
    -------
    selected : ndarray or list of ndarrays
        Selected feature indices. List if n_bootstraps>1.

    test_scores : ndarray or list of ndarrays
        Test statistics. List if n_bootstraps>1.

    threshold : float or list of floats
        Knockoff thresholds. List if n_bootstraps>1.

    X_tildes : ndarray or list of ndarrays
        Generated knockoff variables. List if n_bootstraps>1.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        generator=GaussianDistribution(
            cov_estimator=LedoitWolf(assume_centered=True), random_state=0
        ),
        statistical_test=lasso_statistic_with_sampling,
        centered=True,
        joblib_verbose=0,
        n_repeat=1,
        n_jobs=1,
        memory=None,
    ):
        self.generator = generator
        self.memory = check_memory(memory)
        self.joblib_verbose = joblib_verbose
        self.statistical_test = statistical_test
        self.centered = centered

        assert n_repeat > 0, "n_samplings must be positive"
        self.n_repeat = n_repeat
        # unnecessary to have n_jobs > number of bootstraps
        self.n_jobs = min(n_repeat, n_jobs)

    def fit(self, X, y=None):
        """
        Fit the Model-X Knockoff model by training the generator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data matrix where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,), default=None
            Target values. Not used in this method.

        Returns
        -------
        self : object
            Returns the instance itself.

        Notes
        -----
        The fit method only trains the generator component. The target values y
        are not used in this step.
        """
        if y is not None:
            warnings.warn("y won't be used")
        if self.centered:
            X_ = StandardScaler().fit_transform(X)
        else:
            X_ = X
        self.generator.fit(X_)
        return self

    def _check_fit(self):
        try:
            self.generator._check_fit()
        except ValueError as exc:
            raise ValueError(
                "The Model-X Knockoff requires to be fitted before computing importance"
            ) from exc

    def importance(self, X, y):
        """
        Calculate p-values and identify significant features using the CRT test statistics.

        This function processes the results from Conditional Randomization Test (CRT) to identify
        statistically significant features. It computes p-values by comparing a reference test
        statistic to test statistics from permuted data.

        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

            Array of importance scores (p-values) for each feature. Lower p-values indicate
            higher importance. Values range from 0 to 1.

        Notes
        -----
        The p-values are calculated using the formula:
        (1 + #(T_perm >= T_obs)) / (n_permutations + 1)
        where T_perm are the test statistics from permuted data and T_obs is the
        reference test statistic.

        See Also
        --------
        statistical_test : Method that computes the test statistic used in this function.
        """
        self._check_fit()

        if self.centered:
            X_ = StandardScaler().fit_transform(X)
        else:
            X_ = X

        X_tildes = []
        for i in range(self.n_repeat):
            X_tildes.append(self.generator.sample())

        parallel = Parallel(self.n_jobs, verbose=self.joblib_verbose)
        self.test_scores_ = np.array(
            parallel(
                delayed(self.statistical_test)(X_, X_tildes[i], y)
                for i in range(self.n_repeat)
            )
        )
        self.test_scores_ = np.array(self.test_scores_)

        self.importances_ = np.mean(self.test_scores_, axis=0)
        self.pvalues_ = np.mean(
            [_empirical_pval(self.test_scores_[i]) for i in range(self.n_repeat)],
            axis=0,
        )
        return self.importances_

    def fit_importance(self, X, y, cv=None):
        """
        Fits the model to the data and computes feature importance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data matrix where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            The target values.
        cv : None or cross-validation generator, default=None
            Cross-validation parameter. Not used in this method.
            A warning will be issued if provided.

        Returns
        -------
        importances_ : ndarray of shape (n_features,)
            Feature importance scores (p-values) for each feature.
            Lower values indicate higher importance. Values range from 0 to 1.

        Notes
        -----
        This method combines the fit and importance computation steps.
        It first fits the generator to X and then computes importance scores
        by comparing observed test statistics against permuted ones.

        See Also
        --------
        fit : Method for fitting the generator only
        importance : Method for computing importance scores only
        """
        if cv is not None:
            warnings.warn("cv won't be used")

        self.fit(X)
        return self.importance(X, y)


def model_x_knockoff(
    X,
    y,
    generator=GaussianDistribution(cov_estimator=LedoitWolf(assume_centered=True)),
    statistical_test=lasso_statistic_with_sampling,
    joblib_verbose=0,
    n_repeat=1,
    n_jobs=1,
    memory=None,
):
    model_x_knockoff = ModelXKnockoff(
        generator=generator,
        statistical_test=statistical_test,
        n_repeat=n_repeat,
        n_jobs=n_jobs,
        memory=memory,
        joblib_verbose=joblib_verbose,
    )
    return model_x_knockoff.fit_importance(X, y)


# use the docstring of the class for the function
model_x_knockoff.__doc__ = _aggregate_docstring(
    [
        ModelXKnockoff.__doc__,
        ModelXKnockoff.__init__.__doc__,
        ModelXKnockoff.fit_importance.__doc__,
        ModelXKnockoff.selection.__doc__,
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
