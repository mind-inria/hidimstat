import inspect
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold

from hidimstat._utils.feature_selection import _selection_generic
from hidimstat.base_perturbation import BasePerturbation, BasePerturbationCV
from hidimstat.base_variable_importance import BaseVariableImportance


class SelectorMixin(TransformerMixin):
    """
    Mixin class for feature selection based on different strategies.

    This class defines the fit strategy for all selection methods,
    based on a cross-validation procedure on training data to estimate importance
    values from which p-values are computed for selection.

    .. versionadded:: 0.5.0

    Parameters
    ----------
    estimator: hidimstat-compatible estimator that derives from :class:`hidimstat.BaseVariableImportance`
        The estimator that will be used to perform feature selection. For perturbation-based estimators,
        such as PFI / CFI / LOCI / LOCO, please use the cross-validated alternative (PFICV etc.) for a
        proper feature importance estimation. Automatic fallback to the cross-validated alternative is
        operated with a warning.

    Raise
    -----
    TypeError
        If 'estimator' is not an instance of a hidimstat-compatible estimator.
    UserWarning
        If 'estimator' is an instance of :class:`BasePerturbation` instead of :class:`BasePerturbationCV`.
    """

    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator

    def _check_fit_parameters(self):
        if not isinstance(self.estimator, BaseVariableImportance):
            raise TypeError(
                r"Parameter 'estimator' is not an instance of a "
                "hidimstat-compatible estimator"
            )

        if isinstance(self.estimator, BasePerturbation):
            warnings.warn(
                "An instance of BasePerturbation was given. "
                "This is not supported by the method. Falling back to an "
                "instance of BasePerturbationCV",
                UserWarning,
                stacklevel=2,
            )
            module = __import__("hidimstat")
            signature = inspect.signature(self.estimator.__class__).parameters
            param_dict = {}
            for name in signature:
                param_dict[name] = getattr(self.estimator, name)
            param_dict["estimators"] = param_dict["estimator"]
            del param_dict["estimator"]
            class_ = getattr(
                module, str(self.estimator.__class__.__name__) + "CV"
            )
            self.estimator = class_(cv=StratifiedKFold(), **param_dict)

    def fit(self, X, y):
        self._check_fit_parameters()
        self.estimator.fit_importance(X, y)
        return self


class SelectTopK(SelectorMixin, BaseEstimator):
    """
    Feature selection class from top-k most important features.

    .. versionadded:: 0.5.0

    Parameters
    ----------
    estimator: hidimstat-compatible estimator that derives from :class:`hidimstat.BaseVariableImportance`
        The estimator that will be used to perform feature selection. For perturbation-based estimators,
        such as PFI / CFI / LOCI / LOCO, please use the cross-validated alternative (PFICV etc.) for a
        proper feature importance estimation. Automatic fallback to the cross-validated alternative is
        operated with a warning.
    k_best : int, default=5
        Selects the top k features based on values.

    Raise
    -----
    TypeError
        If 'estimator' is not an instance of a hidimstat-compatible estimator.
    UserWarning
        If 'estimator' is an instance of :class:`BasePerturbation` instead of :class:`BasePerturbationCV`.
    """

    def __init__(self, estimator, k_best=5):
        super().__init__(estimator=estimator)
        self.k_best = k_best

    def transform(self, X):
        if isinstance(self.estimator, BasePerturbationCV):
            # importance shape is gonna be (n_feature_groups, n_folds)
            self.selected_ = _selection_generic(
                self.estimator.importances_.mean(axis=1),
                k_best=self.k_best,
            )
        else:
            self.selected_ = self.estimator.importance_selection(
                k_best=self.k_best
            )

        return X[:, self.selected_]


class PValueSelect(SelectorMixin, BaseEstimator):
    """
    Feature selection class from k-lowest p-values.

    .. versionadded:: 0.5.0

    Parameters
    ----------
    estimator: hidimstat-compatible estimator that derives from :class:`hidimstat.BaseVariableImportance`
        The estimator that will be used to perform feature selection. For perturbation-based estimators,
        such as PFI / CFI / LOCI / LOCO, please use the cross-validated alternative (PFICV etc.) for a
        proper feature importance estimation. Automatic fallback to the cross-validated alternative is
        operated with a warning.
    k_lowest : int, default=None
        Selects the lowest k features based on values.
    percentile : float, default=None
        Selects features based on a specified percentile of values.
        If certain values lie exactly at the given percentile,
        then ties are selected based on a maximum amount of values to return,
        computed as the proportion of total features given by the percentile.
    threshold_max : float, default=None
        Selects features with values below the specified maximum threshold.
    threshold_min : float, default=None
        Selects features with values above the specified minimum threshold.
    alternative_hypothesis : bool, default=False
        If True, selects based on 1-pvalues instead of p-values.

    Raise
    -----
    TypeError
        If 'estimator' is not an instance of a hidimstat-compatible estimator.
    UserWarning
        If 'estimator' is an instance of :class:`BasePerturbation` instead of :class:`BasePerturbationCV`.
    """

    def __init__(
        self,
        estimator,
        k_lowest=None,
        percentile=None,
        threshold_max=0.05,
        threshold_min=None,
        alternative_hypothesis=False,
    ):
        super().__init__(estimator=estimator)
        self.k_lowest = k_lowest
        self.percentile = percentile
        self.threshold_max = threshold_max
        self.threshold_min = threshold_min
        self.alternative_hypothesis = alternative_hypothesis

    def transform(self, X):
        self.selected_ = self.estimator.pvalue_selection(
            k_lowest=self.k_lowest,
            percentile=self.percentile,
            threshold_max=self.threshold_max,
            threshold_min=self.threshold_min,
            alternative_hypothesis=self.alternative_hypothesis,
        )
        return X[:, self.selected_]


class FDRSelect(SelectorMixin, BaseEstimator):
    """
    Feature selection class from False Discovery Rate (FDR) control.

    .. versionadded:: 0.5.0

    Parameters
    ----------
    estimator: hidimstat-compatible estimator that derives from :class:`hidimstat.BaseVariableImportance`
        The estimator that will be used to perform feature selection. For perturbation-based estimators,
        such as PFI / CFI / LOCI / LOCO, please use the cross-validated alternative (PFICV etc.) for a
        proper feature importance estimation. Automatic fallback to the cross-validated alternative is
        operated with a warning.
    fdr : float
        The target false discovery rate level (between 0 and 1)
    fdr_control: {'bhq', 'bhy'}, default='bhq'
        The FDR control method to use:
        - 'bhq': Benjamini-Hochberg procedure
        - 'bhy': Benjamini-Hochberg-Yekutieli procedure
    reshaping_function: callable or None, default=None
        Optional reshaping function for FDR control methods.
        If None, defaults to sum of reciprocals for 'bhy'.
    two_tailed_test: bool, default=False
        If True, performs two-tailed test selection using both p-values
        for positive effects and one-minus p-values for negative effects. The sign
        of the effect is determined from the sign of the importance scores.
    alternative_hypothesis : bool, default=False
        If True, selects based on 1-pvalues instead of p-values.

    Raise
    -----
    TypeError
        If 'estimator' is not an instance of a hidimstat-compatible estimator.
    UserWarning
        If 'estimator' is an instance of :class:`BasePerturbation` instead of :class:`BasePerturbationCV`.
    """

    def __init__(
        self,
        estimator,
        fdr,
        fdr_control="bhq",
        reshaping_function=None,
        two_tailed_test=False,
        alternative_hypothesis=False,
    ):
        super().__init__(estimator=estimator)
        self.fdr = fdr
        self.fdr_control = fdr_control
        self.reshaping_function = reshaping_function
        self.two_tailed_test = two_tailed_test
        self.alternative_hypothesis = alternative_hypothesis

    def transform(self, X):
        self.selected_ = self.estimator.fdr_selection(
            fdr=self.fdr,
            fdr_control=self.fdr_control,
            reshaping_function=self.reshaping_function,
            two_tailed_test=self.two_tailed_test,
        )

        return X[:, self.selected_]


class FWERSelect(SelectorMixin, BaseEstimator):
    """
    Feature selection class from Family-Wise Error Rate (FWER) control.

    .. versionadded:: 0.5.0

    Parameters
    ----------
    estimator: hidimstat-compatible estimator that derives from :class:`hidimstat.BaseVariableImportance`
        The estimator that will be used to perform feature selection. For perturbation-based estimators,
        such as PFI / CFI / LOCI / LOCO, please use the cross-validated alternative (PFICV etc.) for a
        proper feature importance estimation. Automatic fallback to the cross-validated alternative is
        operated with a warning.
    fwer : float
        The target family-wise error rate level (between 0 and 1)
    procedure : {'bonferroni'}, default='bonferroni'
        The FWER control method to use:
        - 'bonferroni': Bonferroni correction
    n_tests : int, default=1
        Factor for multiple testing correction.
    two_tailed_test : bool, default=False
        If True, uses the sign of the importance scores to indicate whether the
        selected features have positive or negative effects.
    alternative_hypothesis : bool, default=False
        If True, selects based on 1-pvalues instead of p-values.

    Raise
    -----
    TypeError
        If 'estimator' is not an instance of a hidimstat-compatible estimator.
    UserWarning
        If 'estimator' is an instance of :class:`BasePerturbation` instead of :class:`BasePerturbationCV`.
    """

    def __init__(
        self,
        estimator,
        fwer,
        procedure="bonferroni",
        n_tests=None,
        two_tailed_test=False,
        alternative_hypothesis=False,
    ):
        super().__init__(estimator=estimator)
        self.fwer = fwer
        self.procedure = procedure
        self.n_tests = n_tests
        self.two_tailed_test = two_tailed_test
        self.alternative_hypothesis = alternative_hypothesis

    def transform(self, X):
        self.selected_ = self.estimator.fwer_selection(
            fwer=self.fwer,
            n_tests=self.n_tests,
            procedure=self.procedure,
            two_tailed_test=self.two_tailed_test,
        )

        return X[:, self.selected_]
