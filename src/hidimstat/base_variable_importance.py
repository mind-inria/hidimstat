import numbers
import warnings
from dataclasses import fields

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, check_is_fitted, clone
from sklearn.exceptions import NotFittedError

from hidimstat._utils.exception import InternalError
from hidimstat._utils.feature_selection import (
    _fdr_selection,
    _fwer_selection,
    _generic_selection,
)
from hidimstat._utils.tags import HidimstatTags
from hidimstat._utils.utils import find_stack_level


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

    def _check_fit_attributes(self):
        """
        Check attributes needed during fitting of the estimator.
        """
        if self.estimator is not None and not isinstance(
            self.estimator, BaseEstimator
        ):
            raise TypeError(
                "'estimator' must be a valid sklearn compatible estimator."
            )

    def _initial_fit(self, estimator: BaseEstimator, *args) -> BaseEstimator:
        """
        Run initial fit of a sklearn estimator.

        Use during fit if an unfitted estimator was passed at instantiation.
        """
        self.importances_ = None
        self.pvalues_ = None

        try:
            check_is_fitted(estimator)
        except NotFittedError:
            print(
                f"Running initial fit of the estimator {estimator.__class__.__name__}."
            )
            return clone(estimator).fit(*args)

        return estimator

    def _check_importance(self):
        """
        Checks if the importance scores have been computed.
        """
        if getattr(self, "importances_", None) is None:
            raise ValueError(
                "The importance method need to be called before calling this method."
            )

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
        return _generic_selection(
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
        if self.pvalues_ is None:
            raise InternalError(
                "The selection on p-value can't be done because the current method does not compute p-values."
            )
        if threshold_max is not None and (
            threshold_max <= 0 or threshold_max >= 1
        ):
            raise ValueError(
                "'threshold_max' strictly needs to be between 0 and 1"
            )
        if threshold_min is not None and (
            threshold_min <= 0 or threshold_min >= 1
        ):
            raise ValueError(
                "'threshold_min' strictly needs to be between 0 and 1"
            )
        if alternative_hypothesis is not None and not isinstance(
            alternative_hypothesis, bool
        ):
            raise ValueError(
                "'alternative_hippothesis' can have only three values: True, False and None."
            )
        return _generic_selection(
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
        selected : ndarray of int of shape (n_features,)
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
        if self.pvalues_ is None:
            raise InternalError(
                "FDR-based selection requires p-values to be computed first. The current method does not support p-values."
            )

        return _fdr_selection(
            p_values=self.pvalues_,
            importances=self.importances_,
            fdr=fdr,
            two_tailed_test=two_tailed_test,
            fdr_control=fdr_control,
            reshaping_function=reshaping_function,
        )

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
        selected : ndarray of int of shape (n_features,)
            Integer array indicating the selected features.
            1 indicates selected features with positive effects,
            -1 indicates selected features with negative effects,
            0 indicates non-selected features.
        """
        self._check_importance()

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

        return _fwer_selection(
            p_values=self.pvalues_,
            importances=self.importances_,
            fwer=fwer,
            n_tests=n_tests,
            procedure=procedure,
            two_tailed_test=two_tailed_test,
        )

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
            if hasattr(self, "feature_groups_"):
                feature_names = list(self.feature_groups_.keys())
            else:
                feature_names = [
                    str(j) for j in range(self.importances_.shape[-1])
                ]
        elif isinstance(feature_names, list):
            if not np.all(isinstance(name, str) for name in feature_names):
                raise ValueError("'feature_names' should be a list of strings")
        else:
            raise ValueError("'feature_names' should be a list")

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

    def __sklearn_tags__(self):
        tags_orig = super().__sklearn_tags__()
        as_dict = {
            field.name: getattr(tags_orig, field.name)
            for field in fields(tags_orig)
        }
        tags = HidimstatTags(**as_dict)
        tags.needs_importance_data = True
        return tags

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "estimator_")


class GroupVariableImportanceMixin:
    """
    Mixin class for adding group functionality to variable importance methods.
    This class provides functionality for handling grouped features in variable
    importance calculations, enabling group-wise selection and importance evaluation.

    Parameters
    ----------
    feature_groups: dict or None, default=None
        Dictionary mapping group names to lists of feature column names/indices.
        If None, each feature is treated as its own group.

    Attributes
    ----------
    n_feature_groups_ : int
        Number of feature groups.
    _feature_groups_ids_ : array-like
        List of feature indices for each group.

    Methods
    -------
    fit(X, y=None)
        Identifies feature groups and validates input data structure.
    _check_compatibility(X)
        Validates compatibility between input data and fitted groups.
    """

    def __init__(self, feature_groups=None):
        self.feature_groups = feature_groups

    def _check_fit_attributes(self):
        """
        Check attributes needed during fitting of the estimator.
        """
        if self.feature_groups is not None and not isinstance(
            self.feature_groups, dict
        ):
            raise TypeError(r"'feature_groups' needs to be a dictionary.")

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
        self._check_fit_attributes()
        del y
        if self.feature_groups is None:
            self.n_feature_groups_ = X.shape[1]
            if isinstance(X, pd.DataFrame):
                self.feature_groups_ = {
                    j: [col] for j, col in enumerate(X.columns)
                }
            else:
                self.feature_groups_ = {
                    j: [j] for j in range(self.n_feature_groups_)
                }
            self._feature_groups_ids_ = np.arange(
                self.n_feature_groups_, dtype=int
            ).reshape(-1, 1)
        elif isinstance(self.feature_groups, dict):
            self.feature_groups_ = self.feature_groups
            self.n_feature_groups_ = len(self.feature_groups_)
            if isinstance(X, pd.DataFrame):
                self._feature_groups_ids_ = []
                for features_group_key in self.feature_groups_:
                    self._feature_groups_ids_.append(
                        [
                            i
                            for i, col in enumerate(X.columns)
                            if col in self.feature_groups_[features_group_key]
                        ]
                    )
            else:
                self._feature_groups_ids_ = [
                    np.array(ids, dtype=int)
                    for ids in list(self.feature_groups_.values())
                ]
        return self

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
        for index_variables in self.feature_groups_.values():
            if isinstance(index_variables[0], numbers.Integral):
                if not np.all(
                    np.array(index_variables, dtype=int) < number_columns
                ):
                    raise ValueError(
                        "X does not correspond to the fitting data."
                    )
            elif type(index_variables[0]) is str or np.issubdtype(
                type(index_variables[0]), str
            ):
                if not np.all([name in names for name in index_variables]):
                    raise ValueError(
                        f"The array is missing at least one of the following columns {index_variables}."
                    )
            else:
                raise InternalError(
                    "A problem with indexing has happened during the fit."
                )
        number_unique_feature_in_groups = np.unique(
            np.concatenate(list(self.feature_groups_.values()))
        ).shape[0]
        if X.shape[1] != number_unique_feature_in_groups:
            warnings.warn(
                f"The number of features in X: {X.shape[1]} differs from the"
                " number of features for which importance is computed: "
                f"{number_unique_feature_in_groups}",
                stacklevel=find_stack_level(),
            )

    def __sklearn_is_fitted__(self) -> bool:
        """
        Built-in sklearn method to check if estimator is fit.
        """
        return (
            getattr(self, "n_feature_groups_", None) is not None
            or getattr(self, "_feature_groups_ids_", None) is not None
        )
