import numbers
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, check_is_fitted, clone
from sklearn.exceptions import NotFittedError

from hidimstat._utils.exception import InternalError
from hidimstat.statistical_tools.multiple_testing import fdr_threshold


def _selection_generic(
    values,
    k_best=None,
    k_lowest=None,
    percentile=None,
    threshold_max=None,
    threshold_min=None,
):
    """
    Helper function for selecting features based on multiple criteria.

    Parameters
    ----------
    values : array-like of shape (n_features,)
        Values to use for feature selection (e.g., importance scores or p-values)
    k_best : int, default=None
        Selects the top k features based on values.
    k_lowest : int, default=None
        Selects the lowest k features based on values.
    percentile : float, default=None
        Selects features based on a specified percentile of values.
    threshold_max : float, default=None
        Selects features with values below the specified maximum threshold.
    threshold_min : float, default=None
        Selects features with values above the specified minimum threshold.

    Returns
    -------
    selection : array-like of shape (n_features,)
        Boolean array indicating the selected features.
    """
    n_criteria = np.sum(
        [
            criteria is not None
            for criteria in [
                k_best,
                k_lowest,
                percentile,
                threshold_max,
                threshold_min,
            ]
        ]
    )
    assert n_criteria <= 1, "Only support selection based on one criteria."
    if k_best is not None:
        assert k_best >= 1, "k_best needs to be positive or None"
        if k_best > values.shape[0]:
            warnings.warn(
                f"k={k_best} is greater than n_features={values.shape[0]}. "
                "All the features will be returned.",
                stacklevel=2,
            )
        mask_k_best = np.zeros_like(values, dtype=bool)

        # based on SelectKBest in Scikit-Learn
        # Request a stable sort. Mergesort takes more memory (~40MB per
        # megafeature on x86-64).
        mask_k_best[np.argsort(values, kind="mergesort")[-k_best:]] = 1
        return mask_k_best
    elif k_lowest is not None:
        assert k_lowest >= 1, "k_lowest needs to be positive or None"
        if k_lowest > values.shape[0]:
            warnings.warn(
                f"k={k_lowest} is greater than n_features={values.shape[0]}. "
                "All the features will be returned.",
                stacklevel=2,
            )
        mask_k_lowest = np.zeros_like(values, dtype=bool)

        # based on SelectKBest in Scikit-Learn
        # Request a stable sort. Mergesort takes more memory (~40MB per
        # megafeature on x86-64).
        mask_k_lowest[np.argsort(values, kind="mergesort")[:k_lowest]] = 1
        return mask_k_lowest
    elif percentile is not None:
        assert 0 < percentile < 100, (
            f"percentile must be between 0 and 100 (exclusive). Got {percentile}."
        )
        # based on SelectPercentile in Scikit-Learn
        threshold_percentile = np.percentile(values, 100 - percentile)
        mask_percentile = values > threshold_percentile
        ties = np.where(values == threshold_percentile)[0]
        if len(ties):
            max_feats = int(len(values) * percentile / 100)
            kept_ties = ties[: max_feats - mask_percentile.sum()]
            mask_percentile[kept_ties] = True
        return mask_percentile
    elif threshold_max is not None:
        mask_threshold_max = values < threshold_max
        return mask_threshold_max
    elif threshold_min is not None:
        mask_threshold_min = values > threshold_min
        return mask_threshold_min
    else:
        no_mask = np.ones_like(values, dtype=bool)
        return no_mask


class BaseVariableImportance(BaseEstimator):
    """
    Base class for variable importance methods.

    This class provides a foundation for implementing variable importance methods,
    including feature selection based on importance scores and p-values.

    Attributes
    ----------
    importances_ : array-like of shape (n_features,), default=None
        The computed importance scores for each feature.
    pvalues_ : array-like of shape (n_features,), default=None
        The computed p-values for each feature.

    Methods
    -------
    selection(k_best=None, percentile=None, threshold=None, threshold_pvalue=None)
        Selects features based on importance scores and/or p-values using various criteria.

    _check_importance()
        Checks if importance scores and p-values have been computed.
    """

    def __init__(self):
        super().__init__()

    def _check_importance(self):
        """
        Checks if the importance scores have been computed.
        """
        if getattr(self, "importances_", None) is None:
            raise ValueError(
                "The importance method need to be called before calling this method."
            )

    def _initial_fit(self, estimator: BaseEstimator, *args) -> BaseEstimator:
        """Run initial fit of a sklearn estimator.

        Use during fit if an unfitted estimator was passed at instantiation.
        """
        self.importances_ = None
        self.pvalues_ = None

        if self.estimator is None:
            raise ValueError(
                "'estimator' must be a valid sklearn compartible estimator."
            )

        if (
            hasattr(estimator, "__sklearn_is_fitted__")
            and not estimator.__sklearn_is_fitted__()
        ):
            print(
                f"Running initial fit of the estimator {estimator.__class__.__name__}."
            )
            return clone(estimator).fit(*args)

        try:
            check_is_fitted(estimator)
        except NotFittedError:
            print(
                f"Running initial fit of the estimator {estimator.__class__.__name__}."
            )
            return clone(estimator).fit(*args)

        return estimator

    def __sklearn_is_fitted__(self):
        return hasattr(self, "estimator_")

    def importance_selection(
        self,
        k_best=None,
        percentile=None,
        threshold_max=None,
        threshold_min=None,
    ):
        """
        Selects features based on variable importance.

        Parameters
        ----------
        k_best : int, default=None
            Selects the top k features based on importance scores.
        percentile : float, default=None
            Selects features based on a specified percentile of importance scores.
        threshold_max : float, default=None
            Selects features with importance scores below the specified maximum threshold.
        threshold_min : float, default=None
            Selects features with importance scores above the specified minimum threshold.

        Returns
        -------
        selection : array-like of shape (n_features,)
            Binary array indicating the selected features.
        """
        self._check_importance()
        return _selection_generic(
            self.importances_,
            k_best=k_best,
            percentile=percentile,
            threshold_max=threshold_max,
            threshold_min=threshold_min,
        )

    def pvalue_selection(
        self,
        k_lowest=None,
        percentile=None,
        threshold_max=0.05,
        threshold_min=None,
        alternative_hypothesis=False,
    ):
        """
        Selects features based on p-values.

        Parameters
        ----------
        k_lowest : int, default=None
            Selects the k features with lowest p-values.
        percentile : float, default=None
            Selects features based on a specified percentile of p-values.
        threshold_max : float, default=0.05
            Selects features with p-values below the specified maximum threshold (0 to 1).
        threshold_min : float, default=None
            Selects features with p-values above the specified minimum threshold (0 to 1).
        alternative_hypothesis : bool, default=False
            If True, selects based on 1-pvalues instead of p-values.

        Returns
        -------
        selection : array-like of shape (n_features,)
            Binary array indicating the selected features (True for selected).
        """
        self._check_importance()
        assert self.pvalues_ is not None, (
            "The selection on p-value can't be done because the current method does not compute p-values."
        )
        if threshold_min is not None:
            assert threshold_min > 0 and threshold_min < 1, (
                "threshold_min needs to be between 0 and 1"
            )
        if threshold_max is not None:
            assert threshold_max > 0 and threshold_max < 1, (
                "threshold_max needs to be between 0 and 1"
            )
        assert alternative_hypothesis is None or isinstance(
            alternative_hypothesis, bool
        ), (
            "alternative_hippothesis can have only three values: True, False and None."
        )
        return _selection_generic(
            self.pvalues_ if not alternative_hypothesis else 1 - self.pvalues_,
            k_lowest=k_lowest,
            percentile=percentile,
            threshold_max=threshold_max,
            threshold_min=threshold_min,
        )

    def fdr_selection(
        self,
        fdr,
        fdr_control="bhq",
        reshaping_function=None,
        two_tailed_test=False,
    ):
        """
        Performs feature selection based on False Discovery Rate (FDR) control.

        Parameters
        ----------
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

        Returns
        -------
        selected : ndarray of int
            Integer array indicating the selected features.
            1 indicates selected features with positive effects,
            -1 indicates selected features with negative effects,
            0 indicates non-selected features.

        Raises
        ------
        ValueError
            If `importances_` haven't been computed yet
        AssertionError
            If `pvalues_` are missing or fdr_control is invalid
        """
        self._check_importance()
        assert fdr > 0 and fdr < 1, "FDR needs to be between 0 and 1 excluded"
        assert self.pvalues_ is not None, (
            "FDR-based selection requires p-values to be computed first. The current method does not support p-values."
        )
        assert fdr_control in {"bhq", "bhy"}, (
            "only 'bhq' and 'bhy' are supported"
        )

        # Adjust fdr for two-tailed test
        if two_tailed_test:
            fdr = fdr / 2

        threshold_pvalues = fdr_threshold(
            self.pvalues_,
            fdr=fdr,
            method=fdr_control,
            reshaping_function=reshaping_function,
        )
        selected = (self.pvalues_ <= threshold_pvalues).astype(int)

        # For two-tailed test, determine the sign of the effect
        if two_tailed_test:
            if self.importances_.ndim > 1:
                sign_beta = np.sign(self.importances_.sum(axis=1))
            else:
                sign_beta = np.sign(self.importances_)
            selected = selected * sign_beta

        return selected

    def fwer_selection(
        self, fwer, procedure="bonferroni", n_tests=None, two_tailed_test=False
    ):
        """
        Performs feature selection based on Family-Wise Error Rate (FWER) control.

        Parameters
        ----------
        fwer : float
            The target family-wise error rate level (between 0 and 1)
        procedure : {'bonferroni'}, default='bonferroni'
            The FWER control method to use:
            - 'bonferroni': Bonferroni correction
        n_tests : int or None, default=None
            Factor for multiple testing correction. If None, uses the number of clusters
            or the number of features in this order.
        two_tailed_test : bool, default=False
            If True, uses the sign of the importance scores to indicate whether the
            selected features have positive or negative effects.

        Returns
        -------
        selected : ndarray of int
            Integer array indicating the selected features.
            1 indicates selected features with positive effects,
            -1 indicates selected features with negative effects,
            0 indicates non-selected features.
        """
        self._check_importance()

        if procedure == "bonferroni":
            if n_tests is None:
                if hasattr(self, "clustering_"):
                    print(
                        "Using number of clusters for multiple testing correction."
                    )
                    n_tests = self.clustering_.n_clusters_
                else:
                    print(
                        "Using number of features for multiple testing correction."
                    )
                    n_tests = self.importances_.shape[0]

            # Adjust fwer for two-tailed test
            if two_tailed_test:
                fwer = fwer / 2

            threshold_pvalue = fwer / n_tests
            selected = (self.pvalues_ < threshold_pvalue).astype(int)
            if two_tailed_test:
                sign_beta = np.sign(self.importances_)
                selected = selected * sign_beta
            return selected

        else:
            raise ValueError("Only 'bonferroni' procedure is supported")

    def plot_importance(
        self,
        ax=None,
        ascending=False,
        feature_names=None,
        **seaborn_barplot_kwargs,
    ):
        """
        Plot feature importances as a horizontal bar plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, (default=None)
            Axes object to draw the plot onto, otherwise uses the current Axes.
        ascending: bool, default=False
            Whether to sort features by ascending importance.
        **seaborn_barplot_kwargs : additional keyword arguments
            Additional arguments passed to seaborn.barplot.
            https://seaborn.pydata.org/generated/seaborn.barplot.html

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Axes object with the plot.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError as e:
            raise Exception(
                "You need to install seaborn for using this functionality"
            ) from e

        self._check_importance()

        if ax is None:
            _, ax = plt.subplots()

        if feature_names is None:
            if hasattr(self, "features_groups_"):
                feature_names = list(self.features_groups_.keys())
            else:
                feature_names = [
                    str(j) for j in range(self.importances_.shape[-1])
                ]
        elif isinstance(feature_names, list):
            assert np.all(isinstance(name, str) for name in feature_names), (
                "The feature_names should be a list of the string"
            )
        else:
            raise ValueError("feature_names should be a list")

        if self.importances_.ndim == 2:
            df_plot = {
                "Feature": feature_names * self.importances_.shape[0],
                "Importance": self.importances_.flatten(),
            }
        else:
            df_plot = {
                "Feature": feature_names,
                "Importance": self.importances_,
            }

        df_plot = pd.DataFrame(df_plot)
        # Sort features by decreasing mean importance
        mean_importance = df_plot.groupby("Feature").mean()
        sorted_features = mean_importance.sort_values(
            by="Importance", ascending=ascending
        ).index
        df_plot["Feature"] = pd.Categorical(
            df_plot["Feature"],
            categories=sorted_features,
            ordered=True,
        )
        sns.barplot(
            df_plot,
            x="Importance",
            y="Feature",
            ax=ax,
            **seaborn_barplot_kwargs,
        )
        sns.despine(ax=ax)
        ax.set_ylabel("")
        return ax


class GroupVariableImportanceMixin:
    """
    Mixin class for adding group functionality to variable importance methods.
    This class provides functionality for handling grouped features in variable
    importance calculations, enabling group-wise selection and importance evaluation.

    Parameters
    ----------
    features_groups: dict or None, default=None
        Dictionary mapping group names to lists of feature column names/indices.
        If None, each feature is treated as its own group.

    Attributes
    ----------
    n_features_groups_ : int
        Number of feature groups.
    _features_groups_ids : array-like
        List of feature indices for each group.

    Methods
    -------
    fit(X, y=None)
        Identifies feature groups and validates input data structure.
    _check_fit()
        Verifies if the instance has been fitted.
    _check_compatibility(X)
        Validates compatibility between input data and fitted groups.
    """

    def __init__(self, features_groups=None):
        self.features_groups = features_groups
        self._features_groups_ids = None

    def fit(self, X, y=None):
        """
        Base fit method for perturbation-based methods. Identifies the groups.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        y: array-like of shape (n_samples,)
            Not used, only present for consistency with the sklearn API.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        del y
        if self.features_groups is None:
            self.n_features_groups_ = X.shape[1]
            self.features_groups_ = {
                j: [j] for j in range(self.n_features_groups_)
            }
            self._features_groups_ids = np.array(
                sorted(self.features_groups_.values()), dtype=int
            )
        elif isinstance(self.features_groups, dict):
            self.features_groups_ = self.features_groups
            self.n_features_groups_ = len(self.features_groups_)
            if isinstance(X, pd.DataFrame):
                self._features_groups_ids = []
                for features_group_key in sorted(self.features_groups_.keys()):
                    self._features_groups_ids.append(
                        [
                            i
                            for i, col in enumerate(X.columns)
                            if col in self.features_groups_[features_group_key]
                        ]
                    )
            else:
                self._features_groups_ids = [
                    np.array(ids, dtype=int)
                    for ids in list(self.features_groups_.values())
                ]
        else:
            raise ValueError("features_groups needs to be a dictionary")
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return (
            getattr(self, "n_features_groups_", None) is not None
            or getattr(self, "_features_groups_ids", None) is not None
        )

    def _check_fit(self):
        """
        Check if the instance has been fitted.

        Raises
        ------
        ValueError
            If the class has not been fitted (i.e., if n_features_groups_
            or _features_groups_ids attributes are missing).
        """
        check_is_fitted(self)

    def _check_compatibility(self, X):
        """
        Check compatibility between input data and fitted model.

        Verifies that the input data X matches the structure expected by the fitted model,
        including feature names and dimensions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to validate. Can be pandas DataFrame or numpy array.

        Raises
        ------
        ValueError
            If X is not a pandas DataFrame or numpy array.
            If column names in X don't match those used during fitting.
        AssertionError
            If feature indices are out of bounds.
            If required feature names are missing from X.
        Warning
            If number of features in X differs from number of grouped features.
        """
        if isinstance(X, pd.DataFrame):
            names = list(X.columns)
        elif isinstance(X, np.ndarray) and X.dtype.names is not None:
            names = X.dtype.names
            # transform Structured Array in pandas array for a better manipulation
            X = pd.DataFrame(X)
        elif isinstance(X, np.ndarray):
            names = None
        else:
            raise ValueError(
                "X should be a pandas dataframe or a numpy array."
            )
        number_columns = X.shape[1]
        for index_variables in self.features_groups_.values():
            if isinstance(index_variables[0], numbers.Integral):
                assert np.all(
                    np.array(index_variables, dtype=int) < number_columns
                ), "X does not correspond to the fitting data."
            elif type(index_variables[0]) is str or np.issubdtype(
                type(index_variables[0]), str
            ):
                assert np.all([name in names for name in index_variables]), (
                    f"The array is missing at least one of the following columns {index_variables}."
                )
            else:
                raise InternalError(
                    "A problem with indexing has happened during the fit."
                )
        number_unique_feature_in_groups = np.unique(
            np.concatenate(list(self.features_groups_.values()))
        ).shape[0]
        if X.shape[1] != number_unique_feature_in_groups:
            warnings.warn(
                f"The number of features in X: {X.shape[1]} differs from the"
                " number of features for which importance is computed: "
                f"{number_unique_feature_in_groups}",
                stacklevel=2,
            )
