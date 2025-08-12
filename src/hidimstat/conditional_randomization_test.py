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
    Compute Lasso statistique

    Parameters
    ----------
    X : 2D ndarray (n_samples, n_features)
        The original design matrix.
    y : 1D ndarray (n_samples, )
        The target vector.
    lasso : Lasso estimator, default=LassoCV( n_jobs=None, verbose=0, max_iter=200000, cv=KFold(n_splits=5, shuffle=True, random_state=0), tol=1e-6)
        The Linear estimator use as stastical tests
    n_alphas : int, default=0
        THe number of alpha to test for preconfiguration of Lasso

    Returns
    -------
    statistical test for each features, i.e. lasso coefficients


    Raises
    ------
    TypeError
        Should an implementation of Lasso estimator
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

    The Conditional Randomization Test :footcite:t:`candes2018panning`.

    Parameters
    ----------
    estimator : sklearn.linear_model.LassoCV
        The Lasso estimator to configure.
    n_jobs : int, default=1
        Number of parallel jobs
    joblib_verbose : int, default=0
        Verbosity level for parallel jobs
    problem_type : {'regression', 'classification'}, default='regression'
        Type of prediction problem when using random forest
    random_state : int, default=2022
        Random seed for reproducibility

    Attributes
    ----------


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
        Fit generator.

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

        References
        ----------
        .. footbibliography::
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
        Calculate p-values and identify significant features using the CRT test
        statistics. This function processes the results from CRT to identify
        statistically significant features while controlling for false discoveries.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            Testing data matrix where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,), default=None
            Target values of testing dataset.

        Returns
        -------
        importances_ : ndarray of shape (n_features,)
            Importance scores for all features
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
