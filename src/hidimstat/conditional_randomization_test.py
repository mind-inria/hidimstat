import warnings

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.utils.validation import check_memory
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

from hidimstat._utils.docstring import _aggregate_docstring
from hidimstat.gaussian_knockoff import (
    GaussianGenerator,
)
from hidimstat.base_variable_importance import BaseVariableImportance


def lasso_statistic(
    X,
    y,
    lasso=LassoCV(
        n_jobs=None,
        verbose=0,
        max_iter=200000,
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        tol=1e-6,
    ),
    n_alphas=0,
):
    """
    Compute Lasso statistic using feature coefficients.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data matrix.
    y : array-like of shape (n_samples,)
        The target values.
    lasso : estimator, default=LassoCV(n_jobs=None, verbose=0, max_iter=200000, cv=KFold(n_splits=5, shuffle=True, random_state=0), tol=1e-6)
        The Lasso estimator to use for computing the test statistic.
    n_alphas : int, default=0
        Number of alpha values to test for Lasso regularization path.
        If 0, uses the default alpha sequence from the estimator.

    Returns
    -------
    coef : ndarray
        Lasso coefficients for each feature.

    Raises
    ------
    TypeError
        If the provided estimator does not have coef_ attribute or is not linear.
    """
    if n_alphas != 0:
        alpha_max = np.max(np.dot(X.T, y)) / (X.shape[1])
        alphas = np.linspace(alpha_max * np.exp(-n_alphas), alpha_max, n_alphas)
        lasso.alphas = alphas
    lasso.fit(X, y)
    if hasattr(lasso, "coef_"):
        coef = np.ravel(lasso.coef_)
    elif hasattr(lasso, "best_estimator_") and hasattr(lasso.best_estimator_, "coef_"):
        coef = np.ravel(lasso.best_estimator_.coef_)  # for CV object
    else:
        raise TypeError("estimator should be linear")
    return coef


class CRT(BaseVariableImportance):
    """
    Implements conditional randomization test (CRT).

    The Conditional Randomization Test :footcite:t:`candes2018panning` is a method
    for statistical variable importance testing.

    Parameters
    ----------
    generator : object, default=GaussianGenerator(cov_estimator=LedoitWolf(assume_centered=True))
        Generator object for simulating null distributions
    statistical_test : callable, default=lasso_statistic
        Function that computes test statistic
    n_permutation : int, default=10
        Number of permutations for the test
    n_jobs : int, default=1
        Number of parallel jobs
    memory : str or object, default=None
        Used for caching
    joblib_verbose : int, default=0
        Verbosity level for parallel jobs
    problem_type : {'regression', 'classification'}, default='regression'
        Type of prediction problem
    random_state : int, default=2022
        Random seed for reproducibility

    Attributes
    ----------
    importances_ : ndarray of shape (n_features,)
        Feature importance scores
    pvalues_ : ndarray of shape (n_features,)
        P-values for each feature

    Notes
    -----
    The CRT tests feature importance by comparing observed test statistics against
    a conditional null distribution generated through simulation.

    See Also
    --------
    GaussianGenerator : Generator for Gaussian null distributions
    lasso_statistic : Default test statistic using Lasso coefficients

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        generator=GaussianGenerator(cov_estimator=LedoitWolf(assume_centered=True)),
        statistical_test=lasso_statistic,
        n_permutation=10,
        n_jobs=1,
        memory=None,
        joblib_verbose=0,
        problem_type="regression",
        random_state=2022,
    ):
        self.generator = generator
        self.n_permutation = n_permutation
        self.n_jobs = n_jobs
        memory = check_memory(memory)
        self.joblib_verbose = joblib_verbose
        self.problem_type = problem_type
        self.random_state = random_state
        self.statistical_test = statistical_test

    def fit(self, X, y=None):
        """
        Fit the CRT model by training the generator.

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

        self.generator.fit(X)
        return self

    def _check_fit(self):
        try:
            self.generator._check_fit()
        except ValueError as exc:
            raise ValueError(
                "The CRT requires to be fitted before computing importance"
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
        reference_value = self.statistical_test(X, y)
        tests = []
        for k in range(self.n_permutation):
            tests.append(self.statistical_test(self.generator.simulate(), y))

        self.pvalues_ = (1 + np.sum(reference_value >= np.array(tests), axis=0)) / (
            self.n_permutation + 1
        )
        self.importances_ = self.pvalues_
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


def crt(
    X,
    y,
    generator=GaussianGenerator(cov_estimator=LedoitWolf(assume_centered=True)),
    statistical_test=lasso_statistic,
    n_permutation=10,
    n_jobs=1,
    memory=None,
    joblib_verbose=0,
    problem_type="regression",
    random_state=2022,
):
    crt = CRT(
        generator=generator,
        statistical_test=statistical_test,
        n_permutation=n_permutation,
        n_jobs=n_jobs,
        memory=memory,
        joblib_verbose=joblib_verbose,
        problem_type=problem_type,
        random_state=random_state,
    )
    return crt.fit_importance(X, y)


# use the docstring of the class for the function
crt.__doc__ = _aggregate_docstring(
    [
        CRT.__doc__,
        CRT.__init__.__doc__,
        CRT.fit_importance.__doc__,
        CRT.selection.__doc__,
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
