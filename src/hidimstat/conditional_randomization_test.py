from copy import deepcopy
from itertools import product
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.covariance import LedoitWolf
from sklearn.utils.validation import check_memory
from tqdm import tqdm

from hidimstat._utils.docstring import _aggregate_docstring
from hidimstat.statistical_tools.gaussian_knockoff import GaussianGenerator
from hidimstat.statistical_tools.lasso_test import lasso_statistic
from hidimstat.base_variable_importance import BaseVariableImportance, SelectionFDR


class CRT(BaseVariableImportance, SelectionFDR):
    """
    Implements conditional randomization test (CRT).

    The Conditional Randomization Test :footcite:t:`candes2018panning` is a method
    for statistical variable importance testing (see algorithm 2).

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
        n_sampling=10,
        n_jobs=1,
        memory=None,
        joblib_verbose=0,
        problem_type="regression",
    ):
        self.generator = generator
        self.n_sampling = n_sampling
        self.n_jobs = n_jobs
        self.memory = check_memory(memory)
        self.joblib_verbose = joblib_verbose
        self.problem_type = problem_type
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

        parallel = Parallel(self.n_jobs, verbose=self.joblib_verbose)
        X_samples = []
        for i in range(self.n_sampling):
            X_samples.append(self.generator.simulate())

        self.test_scores_ = np.array(
            parallel(
                delayed(joblib_statitistic_test)(
                    index, X, X_sample, y, self.statistical_test
                )
                for X_sample, index in tqdm(product(X_samples, range(X.shape[1])))
            )
        )
        self.test_scores_ = reference_value - np.array(self.test_scores_).reshape(
            self.n_sampling, -1
        )

        self.importances_ = np.mean(np.abs(self.test_scores_), axis=0)
        # see equation of p-value in algorithm 2
        self.pvalues_ = (
            1
            + np.sum(
                self.test_scores_ >= 0,
                axis=0,
            )
        ) / (self.n_sampling + 1)
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


def joblib_statitistic_test(index, X, X_sample, y, statistic_test):
    """Compute test statistic for a single feature with permuted data.

    Parameters
    ----------
    index : int
        Index of the feature to test
    X : array-like of shape (n_samples, n_features)
        Original input data matrix
    X_sample : array-like of shape (n_samples, n_features)
        Permuted data matrix
    y : array-like of shape (n_samples,)
        Target values
    statistic_test : callable
        Function that computes the test statistic

    Returns
    -------
    float
        Test statistic value for the specified feature
    """
    X_tmp = deepcopy(X)
    X_tmp[:, index] = deepcopy(X_sample[:, index])
    return statistic_test(X_tmp, y)[index]


def crt(
    X,
    y,
    generator=GaussianGenerator(cov_estimator=LedoitWolf(assume_centered=True)),
    statistical_test=lasso_statistic,
    n_sampling=10,
    n_jobs=1,
    memory=None,
    joblib_verbose=0,
    problem_type="regression",
):
    crt = CRT(
        generator=generator,
        statistical_test=statistical_test,
        n_sampling=n_sampling,
        n_jobs=n_jobs,
        memory=memory,
        joblib_verbose=joblib_verbose,
        problem_type=problem_type,
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
