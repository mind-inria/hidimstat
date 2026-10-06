import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold

from hidimstat.base_variable_importance import _selection_generic
from hidimstat.statistical_tools import nadeau_bengio_ttest
from hidimstat.statistical_tools.multiple_testing import (
    fdr_threshold,
)


class SelectorMixin(TransformerMixin):
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
        importances = self.importances_.mean(axis=0)
        if self.k_best < 0:
            raise ValueError(r"'self.k_best' can't be a negative number.")

        k_best = min(self.k_best, X.shape[1])
        selected = _selection_generic(values=importances, k_best=k_best)
        return X[:, selected]


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
        if self.k_lowest < 0:
            raise ValueError(r"'self.k_lowest' can't be a negative number.")

        k_lowest = min(self.k_lowest, X.shape[1])
        _, p_values = nadeau_bengio_ttest(
            self.importances_,
            pop_mean=0,
            test_frac=1 / (self.cv.get_n_splits() - 1),
            alternative=self.alternative_hypothesis,
        )
        selected = _selection_generic(
            values=p_values,
            k_lowest=k_lowest,
            percentile=self.percentile,
            threshold_max=self.threshold_max,
            threshold_min=self.threshold_min,
        )
        return X[:, selected]


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
            pop_mean=0,
            test_frac=1 / (self.cv.get_n_splits() - 1),
            alternative=self.alternative_hypothesis,
        )
        fdr = self.fdr / 2 if self.two_tailed_test else self.fdr

        threshold_pvalues = fdr_threshold(
            p_values,
            fdr=fdr,
            method=self.fdr_control,
            reshaping_function=self.reshaping_function,
        )
        selected = (self.pvalues_ <= threshold_pvalues).astype(int)

        # For two-tailed test, determine the sign of the effect
        if self.two_tailed_test:
            if self.importances_.ndim > 1:
                sign_beta = np.sign(self.importances_.sum(axis=1))
            else:
                sign_beta = np.sign(self.importances_)
            selected = selected * sign_beta

        return X[:, selected]


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
            pop_mean=0,
            test_frac=1 / (self.cv.get_n_splits() - 1),
            alternative=self.alternative_hypothesis,
        )

        if self.procedure == "bonferroni":
            if self.n_tests is None:
                if hasattr(self, "clustering_"):
                    print(
                        "Using number of clusters for multiple testing correction."
                    )
                    self.n_tests = self.clustering_.n_clusters_
                else:
                    print(
                        "Using number of features for multiple testing correction."
                    )
                    self.n_tests = p_values.shape[0]

            # Adjust fwer for two-tailed test
            if self.two_tailed_test:
                self.fwer = self.fwer / 2

            threshold_pvalue = self.fwer / self.n_tests
            selected = (p_values < threshold_pvalue).astype(int)
            if self.two_tailed_test:
                if self.importances_.ndim > 1:
                    sign_beta = np.sign(self.importances_.sum(axis=1))
                else:
                    sign_beta = np.sign(self.importances_)
                selected = selected * sign_beta
            return X[:, selected]

        else:
            raise ValueError("Only 'bonferroni' procedure is supported")
