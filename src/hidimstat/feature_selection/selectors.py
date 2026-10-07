import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold

from hidimstat._utils.selection import _selection_fdr, _selection_fwer
from hidimstat.base_variable_importance import _selection_generic
from hidimstat.statistical_tools import nadeau_bengio_ttest


class SelectorMixin(TransformerMixin):
    """
    Mixin class for feature selection based on different strategies.

    Parameters
    ----------
    estimator: hidimstat-compatible estimator that derives from :class:`hidimstat.BaseVariableImportance`
        The estimator that will be used to perform feature selection.
    cv: cross-validation generator
        A cross-validation generator object (e.g., KFold, StratifiedKFold).
    """

    def __init__(self, estimator, cv=None):
        super().__init__()
        self.estimator = estimator
        self.cv = cv

    def fit(self, X, y):
        if self.cv is None:
            self.cv = StratifiedKFold()

        self.importances_ = np.zeros((0, X.shape[1]))
        for cv_split_train, cv_split_test in self.cv.split(X, y):
            vim = clone(self.estimator).fit(
                X[cv_split_train], y[cv_split_train]
            )
            importances = vim.importance(X[cv_split_test], y[cv_split_test])
            self.importances_ = np.vstack((self.importances_, importances))
        # (n_feature_groups, n_folds)
        self.importances_ = self.importances_.T
        return self


class SelectTopK(SelectorMixin, BaseEstimator):
    """
    Feature selection class from top-k most important features.
    .. versionadded:: 0.5.0
    """

    def __init__(self, estimator, k_best=5, cv=None):
        super().__init__(estimator=estimator, cv=cv)
        self.k_best = k_best

    def transform(self, X):
        importances = self.importances_.mean(axis=1)
        if self.k_best < 0:
            raise ValueError(r"'self.k_best' can't be a negative number.")

        k_best = min(self.k_best, X.shape[1])
        self.selected_ = _selection_generic(values=importances, k_best=k_best)
        return X[:, self.selected_]


class SelectPValue(SelectorMixin, BaseEstimator):
    """
    Feature selection class from k-lowest p-values.
    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        estimator,
        k_lowest=None,
        percentile=None,
        threshold_max=0.05,
        threshold_min=None,
        alternative_hypothesis="greater",
        cv=None,
    ):
        super().__init__(estimator=estimator, cv=cv)
        self.k_lowest = k_lowest
        self.percentile = percentile
        self.threshold_max = threshold_max
        self.threshold_min = threshold_min
        self.alternative_hypothesis = alternative_hypothesis

    def transform(self, X):
        _, p_values = nadeau_bengio_ttest(
            self.importances_,
            popmean=0,
            test_frac=1 / (self.cv.get_n_splits() - 1),
            alternative=self.alternative_hypothesis,
        )
        self.selected_ = _selection_generic(
            values=p_values,
            k_lowest=self.k_lowest,
            percentile=self.percentile,
            threshold_max=self.threshold_max,
            threshold_min=self.threshold_min,
        )
        return X[:, self.selected_]


class SelectFDR(SelectorMixin, BaseEstimator):
    """
    Feature selection class from False Discovery Rate (FDR) control.
    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        estimator,
        fdr,
        fdr_control="bhq",
        reshaping_function=None,
        two_tailed_test=False,
        alternative_hypothesis="greater",
        cv=None,
    ):
        super().__init__(estimator=estimator, cv=cv)
        self.fdr = fdr
        self.fdr_control = fdr_control
        self.reshaping_function = reshaping_function
        self.two_tailed_test = two_tailed_test
        self.alternative_hypothesis = alternative_hypothesis

    def transform(self, X):
        _, p_values = nadeau_bengio_ttest(
            self.importances_,
            popmean=0,
            test_frac=1 / (self.cv.get_n_splits() - 1),
            alternative=self.alternative_hypothesis,
        )

        self.selected_ = _selection_fdr(
            p_values=p_values,
            importances=self.importances_,
            fdr=self.fdr,
            two_tailed_test=self.two_tailed_test,
            reshaping_function=self.reshaping_function,
        )

        return X[:, self.selected_]


class SelectFWER(SelectorMixin, BaseEstimator):
    """
    Feature selection class from Family-Wise Error Rate (FWER) control.
    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        estimator,
        fwer,
        procedure="bonferroni",
        n_tests=None,
        two_tailed_test=False,
        alternative_hypothesis="greater",
        cv=None,
    ):
        super().__init__(estimator=estimator, cv=cv)
        self.fwer = fwer
        self.procedure = procedure
        self.n_tests = n_tests
        self.two_tailed_test = two_tailed_test
        self.alternative_hypothesis = alternative_hypothesis

    def transform(self, X):
        _, p_values = nadeau_bengio_ttest(
            self.importances_,
            popmean=0,
            test_frac=1 / (self.cv.get_n_splits() - 1),
            alternative=self.alternative_hypothesis,
        )

        if self.n_tests is None:
            self.n_tests = p_values.shape[0]

        self.selected_ = _selection_fwer(
            p_values=p_values,
            importances=self.importances_,
            fwer=self.fwer,
            n_tests=self.n_tests,
            procedure=self.procedure,
            two_tailed_test=self.two_tailed_test,
        )

        return X[:, self.selected_]
