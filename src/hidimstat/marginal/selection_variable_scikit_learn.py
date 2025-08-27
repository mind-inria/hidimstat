from typing import override
import warnings

from sklearn.feature_selection import (
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)

from hidimstat.base_variable_importance import BaseVariableImportance


class AdapterScikitLearn(BaseVariableImportance):
    """
    Adapter base class for scikit-learn feature selection methods.
    This class provides a unified interface for scikit-learn feature selection
    methods to be used within the hidimstat framework.
    Notes
    -----
    Subclasses should implement the `importance` methods.
    """

    def fit(self, X=None, y=None):
        """
        Fit the feature selection model to the data.
        This method does nothing because fitting is not required for these
        scikit-learn feature selection methods.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), optional
            (not used) Input data matrix.
        y : array-like of shape (n_samples,), optional
            (not used) Target values.
        Returns
        -------
        self : object
            Returns self.
        """
        if X is not None:
            warnings.warn("X won't be used")
        if y is not None:
            warnings.warn("y won't be used")
        return self

    def importance(self, X, y):
        """
        Return the computed feature importances.
        This method should be implemented by subclasses to compute feature
        importances for the given data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data matrix.
        y : array-like of shape (n_samples,)
            Target values.
        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError()

    def fit_importance(self, X, y, cv=None):
        """
        Fit the model and compute feature importances.
        This method fits the model (if necessary) and computes feature
        importances for the given data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data matrix.
        y : array-like of shape (n_samples,)
            Target values.
        cv : None or int, optional
            (not used) Cross-validation parameter.
        Returns
        -------
        importances_ : ndarray
            Feature importance scores.
        """
        if cv is not None:
            warnings.warn("cv won't be used")
        self.fit()
        return self.importance(X, y)


class AnalysisOfVariance(AdapterScikitLearn):
    """
    Analysis of Variance (ANOVA) :footcite:t:`fisher1970statistical` feature
    selection for classification tasks.
    This class uses scikit-learn's f_classif to compute F-statistics and p-values
    for each feature. For a short summary of this method, see
    :footcite:t:`larson2008analysis`.
    Attributes
    ----------
    importances_ : ndarray
        1 - p-values for each feature (higher is more important).
    pvalues_ : ndarray
        1 - p-values for each feature.
    f_statitstic_ : ndarray
        F-statistics for each feature.
    Notes
    -----
    See sklearn.feature_selection.f_classif
    """

    def __init__(self):
        super().__init__()

    @override
    def importance(self, X, y):
        """
        Compute ANOVA F-statistics and p-values for each feature.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data matrix.
        y : array-like of shape (n_samples,)
            Target class labels.
        Sets
        ----
        importances_ : ndarray
            1 - p-values for each feature.
        pvalues_ : ndarray
            1 - p-values for each feature.
        f_statitstic_ : ndarray
            F-statistics for each feature.
        Returns
        -------
        importances_ : ndarray
            1 - p-values for each feature.
        """
        f_statistic, p_values = f_classif(X, y)
        # Test the opposite hypothese to the anova
        # Test the similarity in the distribution instead of the difference
        self.importances_ = 1 - p_values
        self.pvalues_ = 1 - p_values
        self.f_statitstic_ = f_statistic
        return self.importances_


class UnivariateLinearRegressionTests(AdapterScikitLearn):
    """
    Univariate linear regression F-test for regression tasks.
    This test is also known as the Chow test :footcite:t:`chow1960tests`.
    Uses scikit-learn's f_regression to compute F-statistics and p-values for each feature.
    Parameters
    ----------
    center : bool, default=True
        If True, center the data before computing F-statistics.
    force_finite : bool, default=True
        If True, replace NaNs and infs in the output with finite numbers.
    Attributes
    ----------
    importances_ : ndarray
        1 - p-values for each feature.
    pvalues_ : ndarray
        1 - p-values for each feature.
    f_statitstic_ : ndarray
        F-statistics for each feature.
    Notes
    -----
    See sklearn.feature_selection.f_regression
    """

    def __init__(self, center=True, force_finite=True):

        super().__init__()
        self.center = center
        self.force_finite = force_finite

    @override
    def importance(self, X, y):
        """
        Compute univariate linear regression F-statistics and p-values for each feature.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data matrix.
        y : array-like of shape (n_samples,)
            Target values.
        Sets
        ----
        importances_ : ndarray
            1 - p-values for each feature.
        pvalues_ : ndarray
            1 - p-values for each feature.
        f_statitstic_ : ndarray
            F-statistics for each feature.
        Returns
        -------
        importances_ : ndarray
            1 - p-values for each feature.
        """
        f_statistic, p_values = f_regression(
            X, y, center=self.center, force_finite=self.force_finite
        )
        # Test the opposite hypothese to the anova
        # Test the similarity in the distribution instead of the difference
        self.importances_ = 1 - p_values
        self.pvalues_ = 1 - p_values
        self.f_statitstic_ = f_statistic
        return self.importances_


class MutualInformation(AdapterScikitLearn):
    """
    Mutual information feature selection for regression or classification.
    This method was introduced by Shannon :footcite:t:`shannon1948mathematical`.
    For an introduction, see section 2.4 of :footcite:t:`cover1999elements`.
    Parameters
    ----------
    problem_type : {'regression', 'classification'}, default='regression'
        Type of prediction problem.
    discrete_features : 'auto' or array-like, default='auto'
        Indicates which features are discrete.
    n_neighbors : int, default=3
        Number of neighbors to use for MI estimation.
    random_state : int, default=None
        Random seed for reproducibility.
    n_jobs : int, default=1
        Number of parallel jobs.
    Attributes
    ----------
    importances_ : ndarray
        Mutual information scores for each feature.
    pvalues_ : None
        P-values are not computed for mutual information.
    Notes
    -----
    See sklearn.feature_selection.mutual_info_regression
    See sklearn.feature_selection.mutual_info_classification
    """

    def __init__(
        self,
        problem_type="regression",
        discrete_features="auto",
        n_neighbors=3,
        random_state=None,
        n_jobs=1,
    ):
        super().__init__()
        assert (
            problem_type == "regression" or problem_type == "classification"
        ), "the value of problem type should be 'regression' or 'classification'"
        self.problem_type = problem_type
        self.discrete_features = discrete_features
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.n_jobs = n_jobs

    @override
    def importance(self, X, y):
        """
        Compute mutual information scores for each feature.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data matrix.
        y : array-like of shape (n_samples,)
            Target values.
        Sets
        ----
        importances_ : ndarray
            Mutual information scores for each feature.
        pvalues_ : None
            P-values are not computed for mutual information.
        Returns
        -------
        importances_ : ndarray
            Mutual information scores for each feature.
        """
        if self.problem_type == "regression":
            mutual_information = mutual_info_regression(
                X,
                y,
                discrete_features=self.discrete_features,
                n_neighbors=self.n_neighbors,
                copy=True,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        elif self.problem_type == "classification":
            mutual_information = mutual_info_classif(
                X,
                y,
                discrete_features=self.discrete_features,
                n_neighbors=self.n_neighbors,
                copy=True,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        else:
            raise ValueError(
                "the value of problem type should be 'regression' or 'classification'"
            )
        self.importances_ = mutual_information
        self.pvalues_ = None

        return self.importances_
