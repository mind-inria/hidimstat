import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


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
