import numbers
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from hidimstat._utils.exception import InternalError


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
    selections_ : array-like of shape (n_features,), default=None
        Binary mask indicating selected features.

    Methods
    -------
    selection(k_best=None, percentile=None, threshold=None, threshold_pvalue=None)
        Selects features based on importance scores and/or p-values using various criteria.

    _check_importance()
        Checks if importance scores and p-values have been computed.
    """

    def __init__(self):
        super().__init__()
        self.importances_ = None
        self.pvalues_ = None
        self.selections_ = None

    def selection(
        self, k_best=None, percentile=None, threshold=None, threshold_pvalue=None
    ):
        """
        Selects features based on variable importance.
        In case several arguments are different from None,
        the returned  selection is the conjunction of all of them.

        Parameters
        ----------
        k_best : int, optional, default=None
            Selects the top k features based on importance scores.
        percentile : float, optional, default=None
            Selects features based on a specified percentile of importance scores.
        threshold : float, optional, default=None
            Selects features with importance scores above the specified threshold.
        threshold_pvalue : float, optional, default=None
            Selects features with p-values below the specified threshold.

        Returns
        -------
        selection : array-like of shape (n_features,)
            Binary array indicating the selected features.
        """
        self._check_importance()
        if k_best is not None:
            if not isinstance(k_best, str) and k_best > self.importances_.shape[1]:
                warnings.warn(
                    f"k={k_best} is greater than n_features={self.importances_.shape[1]}. "
                    "All the features will be returned."
                )
            assert k_best > 0, "k_best needs to be positive and not null"
        if percentile is not None:
            assert (
                0 < percentile and percentile < 100
            ), "percentile needs to be between 0 and 100"
        if threshold_pvalue is not None:
            assert (
                0 < threshold_pvalue and threshold_pvalue < 1
            ), "threshold_pvalue needs to be between 0 and 1"

        # base on SelectKBest of Scikit-Learn
        if k_best == "all":
            mask_k_best = np.ones(self.importances_.shape, dtype=bool)
        elif k_best == 0:
            mask_k_best = np.zeros(self.importances_.shape, dtype=bool)
        elif k_best is not None:
            mask_k_best = np.zeros(self.importances_.shape, dtype=bool)

            # Request a stable sort. Mergesort takes more memory (~40MB per
            # megafeature on x86-64).
            mask_k_best[np.argsort(self.importances_, kind="mergesort")[-k_best:]] = 1
        else:
            mask_k_best = np.ones(self.importances_.shape, dtype=bool)

        # base on SelectPercentile of Scikit-Learn
        if percentile == 100:
            mask_percentile = np.ones(len(self.importances_), dtype=bool)
        elif percentile == 0:
            mask_percentile = np.zeros(len(self.importances_), dtype=bool)
        elif percentile is not None:
            threshold = np.percentile(self.importances_, 100 - percentile)
            mask_percentile = self.importances_ > threshold
            ties = np.where(self.importances_ == threshold)[0]
            if len(ties):
                max_feats = int(len(self.importances_) * percentile / 100)
                kept_ties = ties[: max_feats - mask_percentile.sum()]
                mask_percentile[kept_ties] = True
        else:
            mask_percentile = np.ones(self.importances_.shape, dtype=bool)

        if threshold is not None:
            mask_threshold = self.importances_ < threshold
        else:
            mask_threshold = np.ones(self.importances_.shape, dtype=bool)

        # base on SelectFpr of Scikit-Learn
        if threshold_pvalue is not None:
            mask_threshold_pvalue = self.pvalues_ < threshold_pvalue
        else:
            mask_threshold_pvalue = np.ones(self.importances_.shape, dtype=bool)

        self.selections_ = (
            mask_k_best & mask_percentile & mask_threshold & mask_threshold_pvalue
        )

        return self.selections_

    def _check_importance(self):
        """
        Checks if the importance scores have been computed.
        """
        if self.importances_ is None:
            raise ValueError(
                "The importances need to be called before calling this method"
            )

    def plot_importance(
        self,
        ax=None,
        ascending=False,
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
        except ImportError:
            raise Exception("You need to install seaborn for using this functionality")

        self._check_importance()

        if ax is None:
            _, ax = plt.subplots()
        feature_names = list(self.groups.keys())

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
            df_plot, x="Importance", y="Feature", ax=ax, **seaborn_barplot_kwargs
        )
        sns.despine(ax=ax)
        ax.set_ylabel("")
        return ax


class GroupVariableImportanceMixin:
    """
    Base class for variable importance methods using feature groups.
    This class extends `BaseVariableImportance` to support variable importance
    methods that operate on groups of features, enabling group-wise selection
    and importance evaluation.

    Parameters
    ----------
    feature_groups: dict or None, default=None
        A dictionary where the keys are the group names and the values are the
        list of column names corresponding to each features group. If None,
        the feature_groups are identified based on the columns of X.
    feature_types: str or list, default="auto"
        The feature type. Supported types include "auto", "continuous", and
        "categorical". If "auto", the type is inferred from the cardinality
        of the unique values passed to the `fit` method.

    Attributes
    ----------
    n_feature_groups_ : int
        The number of feature groups.
    _feature_groups_ids : array-like of shape (n_feature_groups,)
        Lists containing the indices of features from X for each feature group.

    Methods
    -------
    fit(X, y=None, groups=None)
        Identifies and stores feature groups based on input or provided grouping.
    _check_fit(X)
        Checks if the class has been fitted and validates group-feature correspondence.
    """

    def __init__(self, feature_groups=None, feature_types="auto"):
        super().__init__()
        self.feature_groups = feature_groups
        self.feature_types = feature_types
        self.n_feature_groups_ = None
        self._feature_groups_ids = None

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
        if self.feature_groups is None:
            self.n_feature_groups_ = X.shape[1]
            self.feature_groups = {j: [j] for j in range(self.n_feature_groups_)}
            self._feature_groups_ids = np.array(
                sorted(list(self.feature_groups.values())), dtype=int
            )
        elif isinstance(self.feature_groups, dict):
            self.n_feature_groups_ = len(self.feature_groups)
            self.feature_groups = self.feature_groups
            if isinstance(X, pd.DataFrame):
                self._feature_groups_ids = []
                for feature_group_key in sorted(self.feature_groups.keys()):
                    self._feature_groups_ids.append(
                        [
                            i
                            for i, col in enumerate(X.columns)
                            if col in self.feature_groups[feature_group_key]
                        ]
                    )
            else:
                self._feature_groups_ids = [
                    np.array(ids, dtype=int)
                    for ids in list(self.feature_groups.values())
                ]
        else:
            raise ValueError("feature_groups needs to be a dictionary")
        if isinstance(self.feature_types, str):
            if self.feature_types in ["auto", "continuous", "categorical"]:
                self.feature_types = [
                    self.feature_types for _ in range(self.n_feature_groups_)
                ]
            else:
                raise ValueError(
                    "feature_types support only the string 'auto', 'continuous', 'categorical'"
                )
        return self

    def _check_fit(self, X):
        """
        Check if the perturbation method has been properly fitted.

        This method verifies that the perturbation method has been fitted by checking
        if required attributes are set and if the number of features matches
        the feature grouped variables.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to validate against the fitted model.

        Raises
        ------
        ValueError
            If the method has not been fitted (i.e., if n_feature_groups_,
            or _feature_groups_ids attributes are missing).
        AssertionError
            If the number of features in X does not match the total number
            of features in the grouped variables.
        """
        if self.n_feature_groups_ is None or self._feature_groups_ids is None:
            raise ValueError("The class is not fitted.")
        if isinstance(X, pd.DataFrame):
            names = list(X.columns)
        elif isinstance(X, np.ndarray) and X.dtype.names is not None:
            names = X.dtype.names
            # transform Structured Array in pandas array for a better manipulation
            X = pd.DataFrame(X)
        elif isinstance(X, np.ndarray):
            names = None
        else:
            raise ValueError("X should be a pandas dataframe or a numpy array.")
        number_columns = X.shape[1]
        for index_variables in self.feature_groups.values():
            if isinstance(index_variables[0], numbers.Integral):
                assert np.all(
                    np.array(index_variables, dtype=int) < number_columns
                ), "X does not correspond to the fitting data."
            elif type(index_variables[0]) is str or np.issubdtype(
                type(index_variables[0]), str
            ):
                assert np.all(
                    [name in names for name in index_variables]
                ), f"The array is missing at least one of the following columns {index_variables}."
            else:
                raise InternalError(
                    "A problem with indexing has happened during the fit."
                )
        number_unique_feature_in_groups = np.unique(
            np.concatenate([values for values in self.feature_groups.values()])
        ).shape[0]
        if X.shape[1] != number_unique_feature_in_groups:
            warnings.warn(
                f"The number of features in X: {X.shape[1]} differs from the"
                " number of features for which importance is computed: "
                f"{number_unique_feature_in_groups}"
            )
