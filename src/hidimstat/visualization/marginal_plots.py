from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.utils.validation import check_is_fitted

from hidimstat.samplers.conditional_sampling import _check_data_type

from .accumulated_local_effects import (
    _bin_indices,
    _build_quantile_grid,
    _predict_fn,
)


def compute_mplot_1d_continuous(
    estimator,
    X,
    feature_idx,
    grid_resolution="auto",
    percentiles=(0.05, 0.95),
    confidence_interval=True,
    confidence_level=0.95,
):
    X = np.asarray(X)
    x = X[:, feature_idx]

    quantiles = _build_quantile_grid(
        x, grid_resolution=grid_resolution, percentiles=percentiles
    )
    n_bins = len(quantiles) - 1

    if n_bins < 1:
        raise ValueError(
            f"Feature {feature_idx} has fewer than 2 unique quantile edges. Increase grid_resolution or check your data."
        )

    bin_idx = _bin_indices(x, quantiles)
    predictions = _predict_fn(estimator, X)

    bin_counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
    bin_sums = np.bincount(bin_idx, weights=predictions, minlength=n_bins)

    mplot = np.zeros(n_bins, dtype=float)
    non_zero = bin_counts > 0
    mplot[non_zero] = bin_sums[non_zero] / bin_counts[non_zero]

    mplot_err = None
    if confidence_interval:
        sample_means = mplot[bin_idx]
        squared_deviations = (predictions - sample_means) ** 2
        sum_sq_dev = np.bincount(
            bin_idx, weights=squared_deviations, minlength=n_bins
        )

        var_of_mean = np.zeros(n_bins, dtype=float)
        valid_bins = bin_counts > 1
        var_of_mean[valid_bins] = sum_sq_dev[valid_bins] / (
            bin_counts[valid_bins] * (bin_counts[valid_bins] - 1)
        )

        z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        mplot_err = z_score * np.sqrt(var_of_mean)

    return {
        "mplot": mplot,
        "quantiles": quantiles,
        "mplot_err": mplot_err,
    }


def compute_mplot_1d_categorical(
    estimator,
    X,
    feature_idx,
    percentiles=(0.05, 0.95),
    confidence_interval=True,
    confidence_level=0.95,
):
    if (
        not isinstance(percentiles, tuple)
        or len(percentiles) != 2
        or not (0.0 <= percentiles[0] <= percentiles[1] <= 1.0)
    ):
        raise ValueError(
            "'percentiles' must be a tuple of 2 floats in [0, 1] in increasing order"
        )

    X = np.array(X)
    x = X[:, feature_idx]

    if X[:, feature_idx].dtype.kind in "iuf":
        low_bnd = np.percentile(x, percentiles[0] * 100)
        high_bnd = np.percentiles(x, percentiles[1] * 100)

        valid_mask = (x >= low_bnd) and (x <= high_bnd)
        X_filtered = X[valid_mask]
        x_filtered = x[valid_mask]
    else:
        X_filtered = X
        x_filtered = x

    unique_values = np.unique(x_filtered)
    n_values = len(unique_values)

    if n_values < 1:
        raise ValueError(
            f"Feature {feature_idx} has no unique values after filtering. "
            "Check your data or widen your percentiles."
        )

    value_idx = np.digitize(x_filtered, unique_values) - 1
    predictions = _predict_fn(estimator, X_filtered)

    value_counts = np.bincoutn(value_idx, minlength=n_values).astype(float)
    value_sums = np.bincount(
        value_idx, weights=predictions, minlength=n_values
    )

    mplot = np.zeros(n_values, dtype=float)
    non_zero = value_counts > 0
    mplot[non_zero] = value_sums[non_zero] / value_counts[non_zero]

    mplot_err = None
    if confidence_interval:
        sample_means = mplot[value_idx]
        squared_deviations = (predictions - sample_means) ** 2
        sum_sq_dev = np.bincount(
            value_idx, weights=squared_deviations, minlength=n_values
        )

        var_of_mean = np.zeros(n_values, dtype=float)
        valid_values = value_counts > 1
        var_of_mean[valid_values] = sum_sq_dev[valid_values] / (
            value_counts[valid_values] * (value_counts[valid_values] - 1)
        )

        z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        mplot_err = z_score * np.sqrt(var_of_mean)

    return {
        "mplot": mplot,
        "unique_values": unique_values,
        "mplot_err": mplot_err,
    }


def compute_mplot_2d(
    estimator,
    X,
    feature_indices,
    grid_resolution="auto",
    percentiles=(0.05, 0.95),
):
    feature_indices = list(feature_indices)
    if len(feature_indices) != 2:
        raise ValueError(
            "feature_indices must contain exactly two feature indices."
        )

    X = np.asarray(X)
    idx_i, idx_j = feature_indices
    x_i, x_j = X[:, idx_i], X[:, idx_j]

    quantiles_i = _build_quantile_grid(
        x_i, grid_resolution=grid_resolution, percentiles=percentiles
    )
    quantiles_j = _build_quantile_grid(
        x_j, grid_resolution=grid_resolution, percentiles=percentiles
    )

    n_bins_i = len(quantiles_i) - 1
    n_bins_j = len(quantiles_j) - 1

    if n_bins_i < 1:
        raise ValueError(
            f"Feature {idx_i} has fewer than 2 unique quantile edges. Increase grid_resolution or check your data."
        )
    if n_bins_j < 1:
        raise ValueError(
            f"Feature {idx_j} has fewer than 2 unique quantile edges. Increase grid_resolution or check your data."
        )

    bin_idx_i = _bin_indices(x_i, quantiles_i)
    bin_idx_j = _bin_indices(x_j, quantiles_j)

    predictions = _predict_fn(estimator, X)

    flat_bin_idx = bin_idx_i * n_bins_j + bin_idx_j
    n_flat_bins = n_bins_i * n_bins_j

    bin_counts_flat = np.bincount(flat_bin_idx, minlength=n_flat_bins).astype(
        float
    )
    bin_sums_flat = np.bincount(
        flat_bin_idx, weights=predictions, minlength=n_flat_bins
    )

    bin_means_flat = np.zeros(n_flat_bins, dtype=float)
    non_zero = bin_counts_flat > 0
    bin_means_flat[non_zero] = (
        bin_sums_flat[non_zero] / bin_counts_flat[non_zero]
    )

    bin_counts = bin_counts_flat.reshape(n_bins_i, n_bins_j)
    mplot = bin_means_flat.reshape(n_bins_i, n_bins_j)

    return {
        "mplot": mplot,
        "quantiles_i": quantiles_i,
        "quantiles_j": quantiles_j,
    }


class MPlot:
    def __init__(self, estimator, feature_names=None):
        self.estimator = estimator
        self.feature_names = feature_names

    def plot(
        self,
        X,
        features,
        feature_type="auto",
        grid_resolution="auto",
        percentiles=(0.05, 0.95),
        confidence_interval=True,
        confidence_level=0.95,
        cmap="viridis",
        **kwargs,
    ):
        X = np.asarray(X)

        if isinstance(features, (int, np.integer)):
            feature_ids = [features]
            feature_type = _check_data_type(
                data_type=feature_type,
                y=X[:, features],
                categorical_max_cardinality=10,
            )
            if feature_type == "continuous":
                plotting_func = self._plot_1d_continuous
                result = compute_mplot_1d_continuous(
                    self.estimator,
                    X,
                    feature_idx=features,
                    grid_resolution=grid_resolution,
                    percentiles=percentiles,
                    confidence_interval=confidence_interval,
                    confidence_level=confidence_level,
                )
            else:
                plotting_func = self._plot_1d_categorical
                result = compute_mplot_1d_categorical(
                    self.estimator,
                    X,
                    feature_idx=features,
                    percentiles=percentiles,
                    confidence_interval=confidence_interval,
                    confidence_level=confidence_level,
                )
        elif isinstance(features, list) and all(
            isinstance(f, (int, np.integer)) for f in features
        ):
            if len(features) > 2:
                raise ValueError(
                    "Only 1D (single int) and 2D (list of two ints) ALE plots are supported."
                )
            feature_ids = copy(features)
            plotting_func = self._plot_2d
            result = compute_mplot_2d(
                self.estimator,
                X,
                percentiles=percentiles,
                feature_indices=features,
                grid_resolution=grid_resolution,
            )
        else:
            raise TypeError("'features' must be an int or a list of int.")

        if self.feature_names is not None:
            feature_names = [self.feature_names[idx] for idx in feature_ids]
        else:
            feature_names = [f"X{idx}" for idx in feature_ids]

        return plotting_func(
            result,
            X,
            feature_ids=feature_ids,
            feature_names=feature_names,
            cmap=cmap,
            **kwargs,
        )

    @staticmethod
    def _plot_1d_continuous(
        result,
        X,
        feature_ids,
        feature_names,
        cmap=None,
        **kwargs,
    ):
        """Render a 1D continuous M-plot with a marginal density strip."""
        del cmap

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
        sns.lineplot(x=centers, y=result["mplot"], ax=ax_main, **kwargs)
        if result["mplot_err"] is not None:
            ax_main.fill_between(
                centers,
                result["mplot"] - result["mplot_err"],
                result["mplot"] + result["mplot_err"],
                color="b",
                alpha=0.15,
                label="Confidence Interval",
            )
        ax_main.set_xlim(low - margin, high + margin)
        ax_main.set_xlabel(feature_names[0])
        ax_main.set_ylabel("M-plot")

        sns.despine(ax=ax_main)
        plt.tight_layout()
        return axes

    @staticmethod
    def _plot_1d_discrete(
        result,
        X,
        feature_ids,
        feature_names,
        cmap=None,
        **kwargs,
    ):
        """Render a 1D discrete M-plot with a marginal density histogram."""
        del cmap

        feature_values = X[:, feature_ids[0]]

        if result["quantiles"].dtype.kind in "iuf":
            low, high = (
                result["unique_values"].min(),
                result["unique_values"].max(),
            )
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
            x=result["unique_values"],
            y=result["mplot"],
            ax=ax_main,
            marker="o",
            **kwargs,
        )
        if result["mplot_err"] is not None:
            ax_main.fill_between(
                result["unique_values"],
                result["mplot"] - result["mplot_err"],
                result["mplot"] + result["mplot_err"],
                color="b",
                alpha=0.15,
            )
        ax_main.xaxis.set_ticks(result["unique_values"])
        ax_main.set_xlim(low - 0.6, high + 0.6)
        ax_main.set_xlabel(feature_names[0])
        ax_main.set_ylabel("M-plot")

        sns.despine(ax=ax_main)
        plt.tight_layout()
        return axes

    @staticmethod
    def _plot_2d(
        result,
        X,
        feature_ids,
        feature_names,
        cmap="viridis",
        **kwargs,
    ):
        """Render a 2D M-plot with marginal density strips on each axis."""
        x = X[:, feature_ids[0]]
        y = X[:, feature_ids[1]]

        quantiles_i = result["quantiles_i"]
        quantiles_j = result["quantiles_j"]
        mplot = result["mplot"]

        low_i, high_i = quantiles_i.min(), quantiles_i.max()
        low_j, high_j = quantiles_j.min(), quantiles_j.max()

        zz_cells = (
            mplot[:-1, :-1] + mplot[1:, 1:] + mplot[:-1, 1:] + mplot[1:, :-1]
        ) / 4.0

        fig, axes = plt.subplots(
            2,
            5,
            figsize=(8, 6),
            height_ratios=[0.2, 1],
            width_ratios=[1, 0.04, 0.2, 0.2, 0.05],
            gridspec_kw={"wspace": 0, "hspace": 0.08},
        )

        for k in range(1, 5):
            axes[0, k].axis("off")
        axes[1, 1].axis("off")
        axes[1, 3].axis("off")

        ax_main = axes[1, 0]
        mesh = ax_main.pcolormesh(
            quantiles_i,
            quantiles_j,
            zz_cells.T,
            cmap=cmap,
            shading="nearest",
            edgecolors="face",
            **kwargs,
        )

        level_lines = [0.1, 0.3, 0.5, 0.7, 0.9]
        sns.kdeplot(
            x=x,
            y=y,
            ax=ax_main,
            levels=level_lines,
            colors="black",
            linewidths=0.8,
            alpha=0.7,
        )
        cs = ax_main.collections[-1]
        ax_main.clabel(
            cs,
            inline=True,
            fontsize=9,
            colors="black",
            fmt=lambda x_val: (
                f"{level_lines[list(cs.levels).index(x_val)]:.1f}"
            ),
        )

        ax_main.set_xlabel(feature_names[0])
        ax_main.set_ylabel(feature_names[1])
        ax_main.set_xlim(low_i, high_i)
        ax_main.set_ylim(low_j, high_j)

        ax_top = axes[0, 0]
        sns.kdeplot(
            x, ax=ax_top, fill=True, color="black", alpha=0.25, legend=False
        )
        sns.despine(ax=ax_top, left=True)
        ax_top.xaxis.set_ticks([])
        ax_top.yaxis.set_visible(False)
        ax_top.set_xlim(low_i, high_i)

        ax_right = axes[1, 2]
        sns.kdeplot(
            y=y,
            ax=ax_right,
            fill=True,
            color="black",
            alpha=0.25,
            legend=False,
        )
        sns.despine(ax=ax_right, bottom=True)
        ax_right.yaxis.set_ticks([])
        ax_right.xaxis.set_visible(False)
        ax_right.set_ylim(low_j, high_j)

        ax_cbar = axes[1, 4]
        cbar = fig.colorbar(mesh, cax=ax_cbar)
        cbar.set_label("M-plot")

        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.16, top=0.9)
        return axes
