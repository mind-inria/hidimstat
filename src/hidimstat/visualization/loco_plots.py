from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor

from hidimstat._utils.grid import _bin_indices, _build_quantile_grid_1d
from hidimstat.samplers.conditional_sampling import _check_data_type

from .accumulated_local_effects import _predict_fn


def compute_loco_plot_1d_continuous(
    estimator,
    X,
    y,
    feature_idx,
    grid_resolution="auto",
    confidence_interval=True,
    confidence_level=0.95,
    percentiles=(5, 95),
):
    """Compute the 1D LOCO-plot for a single continuous feature using binning.

    Parameters
    ----------
    estimator : fitted sklearn-compatible estimator
        Must expose `predict`, `predict_proba`, or `decision_function`.
    X : array-like of shape (n_samples, n_features)
        Training (or evaluation) dataset.
    feature_idx : int
        Column index of the feature of interest.
    grid_resolution : int or "auto", default="auto"
        Number of bins used to build the quantile grid with continuous features.

        - If "auto", the number of bins is determined automatically
          to minimize the histogram error.
        - Note that the final number of bins in the quantile grid may be
          strictly less than `grid_resolution` (or the auto-calculated value)
          if the data contains many duplicate values or fewer unique points
          than requested.

    confidence_interval : bool, default=True
        Whether to compute the confidence intervals of the LOCO curve.
    confidence_level : float, default=0.95
        The confidence level used to compute the confidence intervals (e.g., 0.95 for 95%).
    percentiles : tuple of float, default=(5, 95)
        The lower and upper percentile used to create the extreme values for the grid.
        Must be in [0, 100].

    Returns
    -------
    loco_plot : ndarray of shape (n_bins,)
        LOCO-plot values per bin.
    quantiles : ndarray of shape (n_quantiles,)
        Bin edges.
    loco_err : ndarray of shape (n_bins,) or None
        The margin of error for each bin at the specified confidence level.
        Returns `None` if `confidence_interval` is False.
    """
    X = np.asarray(X)
    x_j = X[:, feature_idx]

    # Grid defined by quantiles
    quantiles = _build_quantile_grid_1d(
        x_j, grid_resolution=grid_resolution, percentiles=percentiles
    )
    n_bins = len(quantiles) - 1

    if n_bins < 1:
        raise ValueError(
            f"Feature {feature_idx} has fewer than 2 unique quantile edges."
        )

    bin_idx = _bin_indices(x_j, quantiles)

    # Estimator of y with X_{-j}
    estimator_minus_j = HistGradientBoostingRegressor()

    X_minus_j = np.delete(X, feature_idx, axis=1)
    estimator_minus_j.fit(X_minus_j, y)

    # Compute differences \Delta = \mu(X) - \mu_{-j}(X_{-j})
    predictions_real = _predict_fn(estimator, X)
    predictions_minus_j = _predict_fn(estimator_minus_j, X_minus_j)
    individual_deltas = predictions_real - predictions_minus_j

    # Average within each bin
    bin_counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
    bin_sums = np.bincount(
        bin_idx, weights=individual_deltas, minlength=n_bins
    )

    loco_plot = np.zeros(n_bins, dtype=float)
    non_zero = bin_counts > 0
    loco_plot[non_zero] = bin_sums[non_zero] / bin_counts[non_zero]
    loco_plot -= np.sum(loco_plot * bin_counts) / bin_counts.sum()

    # Compute confidence interval
    loco_err = None
    if confidence_interval:
        sample_means = loco_plot[bin_idx]
        squared_deviations = (individual_deltas - sample_means) ** 2
        sum_sq_dev = np.bincount(
            bin_idx, weights=squared_deviations, minlength=n_bins
        )

        var_of_mean = np.zeros(n_bins, dtype=float)
        valid_bins = bin_counts > 1
        var_of_mean[valid_bins] = sum_sq_dev[valid_bins] / (
            bin_counts[valid_bins] * (bin_counts[valid_bins] - 1)
        )

        z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        loco_err = z_score * np.sqrt(var_of_mean)

    return loco_plot, quantiles, loco_err


def compute_loco_plot_1d_categorical(
    estimator,
    X,
    y,
    feature_idx,
    confidence_interval=True,
    confidence_level=0.95,
    percentiles=(5, 95),
):
    """Compute the 1D LOCO-plot for a single categorical feature per unique value.

    Parameters
    ----------
    estimator : fitted sklearn-compatible estimator
        Must expose `predict`, `predict_proba`, or `decision_function`.
    X : array-like of shape (n_samples, n_features)
        Training (or evaluation) dataset.
    feature_idx : int
        Column index of the feature of interest.
    confidence_interval : bool, default=True
        Whether to compute the confidence intervals of the LOCO curve.
    confidence_level : float, default=0.95
        The confidence level used to compute the confidence intervals (e.g., 0.95 for 95%).
    percentiles : tuple of float, default=(5, 95)
        The lower and upper percentile used to create the extreme values for the grid.
        Must be in [0, 100].

    Returns
    -------
    dict
        A dictionary containing:
        - "loco_plot": ndarray of shape (n_values,) - LOCO-plot values per unique value.
        - "unique_values": ndarray of shape (n_values,) - Distinct categories evaluated.
        - "loco_err": ndarray of shape (n_values,) or None - Margin of error per category.
    """
    if (
        not isinstance(percentiles, tuple)
        or len(percentiles) != 2
        or not (0.0 <= percentiles[0] <= percentiles[1] <= 1.0)
    ):
        raise ValueError(
            "'percentiles' must be a tuple of 2 floats in [0, 1] in increasing order"
        )

    X = np.array(X)
    x_j = X[:, feature_idx]

    # Filter extreme values if numerical categories
    if x_j.dtype.kind in "iuf":
        low_bnd = np.percentile(x_j, percentiles[0] * 100)
        high_bnd = np.percentile(x_j, percentiles[1] * 100)

        valid_mask = (x_j >= low_bnd) & (x_j <= high_bnd)
        X_filtered = X[valid_mask]
        x_filtered = x_j[valid_mask]
    else:
        X_filtered = X
        x_filtered = x_j

    unique_values = np.unique(x_filtered)
    n_values = len(unique_values)

    if n_values < 1:
        raise ValueError(
            f"Feature {feature_idx} has no unique values after filtering."
        )

    value_idx = np.digitize(x_filtered, unique_values) - 1

    # Estimator of X_j with X_{-j}
    estimator_minus_j = HistGradientBoostingRegressor()

    X_minus_j = np.delete(X_filtered, feature_idx, axis=1)
    estimator_minus_j.fit(X_minus_j, y)

    # Compute differences \Delta = \mu(X) - \mu_{-j}(X_{-j})
    predictions_real = _predict_fn(estimator, X_filtered)
    predictions_minus_j = _predict_fn(estimator_minus_j, X_minus_j)
    individual_deltas = predictions_real - predictions_minus_j

    # Average within each bin
    value_counts = np.bincount(value_idx, minlength=n_values).astype(float)
    value_sums = np.bincount(
        value_idx, weights=individual_deltas, minlength=n_values
    )

    loco_plot = np.zeros(n_values, dtype=float)
    non_zero = value_counts > 0
    loco_plot[non_zero] = value_sums[non_zero] / value_counts[non_zero]

    # Compute confidence interval
    loco_err = None
    if confidence_interval:
        sample_means = loco_plot[value_idx]
        squared_deviations = (individual_deltas - sample_means) ** 2
        sum_sq_dev = np.bincount(
            value_idx, weights=squared_deviations, minlength=n_values
        )

        var_of_mean = np.zeros(n_values, dtype=float)
        valid_values = value_counts > 1
        var_of_mean[valid_values] = sum_sq_dev[valid_values] / (
            value_counts[valid_values] * (value_counts[valid_values] - 1)
        )

        z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        loco_err = z_score * np.sqrt(var_of_mean)

    return loco_plot, unique_values, loco_err


class LOCOPlot:
    """Leave One Covariate Out (LOCO) visualization.

    LOCO measures how the predictions of a model change on average when a
    feature's true value is compared against its conditionally predicted trend.
    Unlike standard marginal plots (M-plots), LOCO-plots isolate the unique marginal
    contribution of a feature by conditioning out the effect of its correlation with
    other features via an auxiliary generator :math:`\\eta_j(X_{-j})`.

    Parameters
    ----------
    estimator : fitted sklearn-compatible estimator
        Must expose `predict`, `predict_proba`, or `decision_function`.
    feature_names : list of str, optional
        Names of the features. If None, X0, X1, ... will be used.
    """

    def __init__(self, estimator, feature_names=None):
        self.estimator = estimator
        self.feature_names = feature_names

    def plot(
        self,
        X,
        y,
        features,
        feature_type="auto",
        grid_resolution="auto",
        confidence_interval=True,
        confidence_level=0.95,
        percentiles=(5, 95),
        **kwargs,
    ):
        """Compute and display the LOCO plot for a single feature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dataset used to build the quantile grid and fit the generator.
        features : int
            Feature index of the variable of interest.
        feature_type : string among "auto", "continuous", or "categorical"
            Specify the type of values the feature has. Set by default to
            auto and in this case :

            - non-numeric feature : categorical
            - numeric feature : categorical if the feature has less than 10 unique
              values, and continuous otherwise
        grid_resolution : int or "auto", default="auto"
            Number of bins used to build the quantile grid with continuous features.

            - If "auto", the number of bins is determined automatically
            to minimize the histogram error.
            - Note that the final number of bins in the quantile grid may be
            strictly less than `grid_resolution` (or the auto-calculated value)
            if the data contains many duplicate values or fewer unique points
            than requested.
        confidence_interval : bool, default=True
            Whether to compute and display confidence intervals around the curve.
        confidence_level : float, default=0.95
            The confidence level used to compute the confidence intervals (e.g., 0.95 for 95%).
        percentiles : tuple of float, default=(5, 95)
            The lower and upper percentile used to create the extreme values for the grid.
            Must be in [0, 100].
        **kwargs
            Extra keyword arguments forwarded to `sns.lineplot`.
        """
        X = np.asarray(X)

        if not isinstance(features, (int, np.integer)):
            raise TypeError("'features' must be an int.")

        feature_ids = [features]
        feature_type = _check_data_type(
            data_type=feature_type,
            y=X[:, features],
            categorical_max_cardinality=10,
        )

        if feature_type == "continuous":
            plotting_func = self._plot_1d_continuous
            loco_plot, quantiles, loco_err = compute_loco_plot_1d_continuous(
                estimator=self.estimator,
                X=X,
                y=y,
                feature_idx=features,
                grid_resolution=grid_resolution,
                confidence_interval=confidence_interval,
                confidence_level=confidence_level,
                percentiles=percentiles,
            )
            result = {
                "loco_plot": loco_plot,
                "quantiles": quantiles,
                "loco_err": loco_err,
            }
        else:
            plotting_func = self._plot_1d_categorical
            loco_plot, unique_values, loco_err = (
                compute_loco_plot_1d_categorical(
                    estimator=self.estimator,
                    X=X,
                    y=y,
                    feature_idx=features,
                    confidence_interval=confidence_interval,
                    confidence_level=confidence_level,
                    percentiles=percentiles,
                )
            )
            result = {
                "loco_plot": loco_plot,
                "unique_values": unique_values,
                "loco_err": loco_err,
            }

        if self.feature_names is not None:
            feature_names = [self.feature_names[idx] for idx in feature_ids]
        else:
            feature_names = [f"X{idx}" for idx in feature_ids]

        return plotting_func(
            result,
            X,
            feature_ids=feature_ids,
            feature_names=feature_names,
            **kwargs,
        )

    @staticmethod
    def _plot_1d_continuous(
        result,
        X,
        feature_ids,
        feature_names,
        **kwargs,
    ):
        """Render a 1D continuous LOCO-plot with a marginal density strip."""
        feature_values = X[:, feature_ids[0]]
        centers = (result["quantiles"][1:] + result["quantiles"][:-1]) / 2
        low, high = centers.min(), centers.max()
        margin = (high - low) * 0.05

        _, axes = plt.subplots(2, 1, height_ratios=[0.2, 1], sharex=True)

        ax_top = axes[0]
        sns.kdeplot(
            feature_values,
            ax=ax_top,
            fill=True,
            color="black",
            alpha=0.25,
            legend=False,
        )
        sns.despine(ax=ax_top, left=True)
        ax_top.tick_params(axis="x", which="both", bottom=False, top=False)
        ax_top.yaxis.set_visible(False)

        ax_main = axes[1]
        sns.lineplot(x=centers, y=result["loco_plot"], ax=ax_main, **kwargs)
        if result["loco_err"] is not None:
            ax_main.fill_between(
                centers,
                result["loco_plot"] - result["loco_err"],
                result["loco_plot"] + result["loco_err"],
                color="b",
                alpha=0.15,
            )

        ax_main.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax_main.set_xlim(low - margin, high + margin)
        ax_main.set_xlabel(feature_names[0])
        ax_main.set_ylabel("LOCO plot")

        sns.despine(ax=ax_main)
        plt.tight_layout()
        return axes

    @staticmethod
    def _plot_1d_categorical(
        result,
        X,
        feature_ids,
        feature_names,
        **kwargs,
    ):
        """Render a 1D categorical LOCO-plot with a marginal density histogram."""
        feature_values = X[:, feature_ids[0]]
        unique_values = result["unique_values"]

        if unique_values.dtype.kind in "iuf":
            low, high = unique_values.min(), unique_values.max()
            feature_values_filtered = feature_values[
                (feature_values >= low) & (feature_values <= high)
            ]
        else:
            feature_values_filtered = feature_values

        _, axes = plt.subplots(
            2, 1, figsize=(8, 4), height_ratios=[0.2, 1], sharex=True
        )

        ax_top = axes[0]
        sns.histplot(
            feature_values_filtered,
            ax=ax_top,
            discrete=True,
            fill=True,
            color="black",
            alpha=0.25,
            legend=False,
        )
        sns.despine(ax=ax_top, left=True)
        ax_top.tick_params(axis="x", which="both", bottom=False, top=False)
        ax_top.yaxis.set_visible(False)

        ax_main = axes[1]
        sns.lineplot(
            x=unique_values,
            y=result["loco_plot"],
            ax=ax_main,
            marker="o",
            **kwargs,
        )
        if result["loco_err"] is not None:
            ax_main.fill_between(
                unique_values,
                result["loco_plot"] - result["loco_err"],
                result["loco_plot"] + result["loco_err"],
                color="b",
                alpha=0.15,
            )

        ax_main.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax_main.xaxis.set_ticks(unique_values)
        if unique_values.dtype.kind in "iuf":
            ax_main.set_xlim(low - 0.6, high + 0.6)
        ax_main.set_xlabel(feature_names[0])
        ax_main.set_ylabel("LOCO plot")

        sns.despine(ax=ax_main)
        plt.tight_layout()
        return axes
