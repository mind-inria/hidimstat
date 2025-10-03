import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from hidimstat.statistical_tools.multiple_testing import fdr_threshold


def _selection_multi_criteria(
    values, k_best=None, percentile=None, threshold_max=None, threshold_min=None
):
    """
    Helper function for selecting features based on multiple criteria.

    Parameters
    ----------
    values : array-like of shape (n_features,)
        Values to use for feature selection (e.g., importance scores or p-values)
    k_best : int, default=None
        Selects the top k features based on values.
    percentile : float, default=None
        Selects features based on a specified percentile of values.
    threshold_max : float, default=None
        Selects features with values below the specified maximum threshold.
    threshold_min : float, default=None
        Selects features with values above the specified minimum threshold.

    Returns
    -------
    selections : array-like of shape (n_features,)
        Boolean array indicating the selected features.
    """
    if k_best is not None:
        assert k_best >= 1, "k_best needs to be positive or None"
        if k_best > values.shape[0]:
            warnings.warn(
                f"k={k_best} is greater than n_features={values.shape[0]}. "
                "All the features will be returned."
            )
    if percentile is not None:
        assert (
            0 < percentile < 100
        ), "percentile must be between 0 and 100 (exclusive). Got {}.".format(
            percentile
        )
    if threshold_max is not None and threshold_min is not None:
        assert (
            threshold_max > threshold_min
        ), "threshold_max needs to be higher than threshold_min "

    # based on SelectKBest in Scikit-Learn
    if k_best is not None:
        mask_k_best = np.zeros_like(values, dtype=bool)

        # Request a stable sort. Mergesort takes more memory (~40MB per
        # megafeature on x86-64).
        mask_k_best[np.argsort(values, kind="mergesort")[-k_best:]] = 1
    else:
        mask_k_best = np.ones_like(values, dtype=bool)

    # based on SelectPercentile in Scikit-Learn
    if percentile is not None:
        threshold_percentile = np.percentile(values, 100 - percentile)
        mask_percentile = values > threshold_percentile
        ties = np.where(values == threshold_percentile)[0]
        if len(ties):
            max_feats = int(len(values) * percentile / 100)
            kept_ties = ties[: max_feats - mask_percentile.sum()]
            mask_percentile[kept_ties] = True
    else:
        mask_percentile = np.ones_like(values, dtype=bool)

    if threshold_max is not None:
        mask_threshold_max = values < threshold_max
    else:
        mask_threshold_max = np.ones_like(values, dtype=bool)

    if threshold_min is not None:
        mask_threshold_min = values > threshold_min
    else:
        mask_threshold_min = np.ones_like(values, dtype=bool)

    selections = mask_k_best & mask_percentile & mask_threshold_max & mask_threshold_min
    return selections


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
        self.importances_ = None
        self.pvalues_ = None

    def _check_importance(self):
        """
        Checks if the importance scores have been computed.
        """
        if self.importances_ is None:
            raise ValueError(
                "The importances need to be called before calling this method"
            )

    def importance_selection(
        self, k_best=None, percentile=None, threshold_max=None, threshold_min=None
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
        return _selection_multy_criteria(
            self.importances_,
            k_best=k_best,
            percentile=percentile,
            threshold_max=threshold_max,
            threshold_min=threshold_min,
        )

    def pvalue_selection(
        self,
        k_best=None,
        percentile=None,
        threshold_max=0.05,
        threshold_min=None,
        alternative_hypothesis=False,
    ):
        """
        Selects features based on p-values.

        Parameters
        ----------
        k_best : int, default=None
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
        assert (
            self.pvalues_ is not None
        ), "The selection on p-value can't be done because the current method does not compute p-values."
        if threshold_min is not None:
            assert (
                0 < threshold_min and threshold_min < 1
            ), "threshold_min needs to be between 0 and 1"
        if threshold_max is not None:
            assert (
                0 < threshold_max and threshold_max < 1
            ), "threshold_max needs to be between 0 and 1"
        assert alternative_hypothesis is None or isinstance(
            alternative_hypothesis, bool
        ), "alternative_hippothesis can have only three values: True, False and None."
        return _selection_multy_criteria(
            self.pvalues_ if not alternative_hypothesis else 1 - self.pvalues_,
            k_best=k_best,
            percentile=percentile,
            threshold_max=threshold_max,
            threshold_min=threshold_min,
        )

    def fdr_selection(
        self,
        fdr,
        fdr_control="bhq",
        reshaping_function=None,
        alternative_hypothesis=False,
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
        alternative_hippothesis: bool or None, default=False
            If False, selects features with small p-values.
            If True, selects features with large p-values (close to 1).
            If None, selects features that have either small or large p-values.

        Returns
        -------
        selected : ndarray of bool
            Boolean mask of selected features.
            True indicates selected features, False indicates non-selected features.

        Raises
        ------
        ValueError
            If `importances_` haven't been computed yet
        AssertionError
            If `pvalues_` are missing or fdr_control is invalid
        """
        self._check_importance()
        assert 0 < fdr and fdr < 1, "FDR needs to be between 0 and 1 excluded"
        assert (
            self.pvalues_ is not None
        ), "FDR-based selection requires p-values to be computed first. The current method does not support p-values."
        assert (
            fdr_control == "bhq" or fdr_control == "bhy"
        ), "only 'bhq' and 'bhy' are supported"
        assert alternative_hypothesis is None or isinstance(
            alternative_hypothesis, bool
        ), "alternative_hippothesis can have only three values: True, False and None."

        # selection on pvalue
        if alternative_hypothesis is None or not alternative_hypothesis:
            threshold_pvalues = fdr_threshold(
                self.pvalues_,
                fdr=fdr,
                method=fdr_control,
                reshaping_function=reshaping_function,
            )
            selected_pvalues = self.pvalues_ <= threshold_pvalues
        else:
            selected_pvalues = np.zeros_like(self.pvalues_, dtype=bool)

        # selection on 1-pvalue
        if alternative_hypothesis is None or alternative_hypothesis:
            threshold_one_minus_pvalues = fdr_threshold(
                1 - self.pvalues_,
                fdr=fdr,
                method=fdr_control,
                reshaping_function=reshaping_function,
            )
            selected_one_minus_pvalues = (
                1 - self.pvalues_
            ) <= threshold_one_minus_pvalues
        else:
            selected_one_minus_pvalues = np.zeros_like(self.pvalues_, dtype=bool)

        selected = selected_pvalues | selected_one_minus_pvalues
        return selected

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
