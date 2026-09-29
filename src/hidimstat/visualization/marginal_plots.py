import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.utils.validation import check_is_fitted

from hidimstat._utils.grid import _bin_indices, _build_quantile_grid_1d
from hidimstat.samplers.conditional_sampling import _check_data_type

from .accumulated_local_effects import _predict_fn


def _compute_bin_mean_and_variance(
    predictions,
    n_bins,
    bin_indices,
    confidence_interval=True,
    confidence_level: float = 0.95,
) -> np.ndarray | None:
    """Compute variance of the mean and confidence interval width for binned predictions."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1.")

    bin_counts = np.bincount(bin_indices, minlength=n_bins).astype(float)
    bin_sums = np.bincount(bin_indices, weights=predictions, minlength=n_bins)

    bin_means = np.zeros(n_bins, dtype=float)
    non_zero = bin_counts > 0
    bin_means[non_zero] = bin_sums[non_zero] / bin_counts[non_zero]

    bin_vars = None
    if confidence_interval:
        valid_bins = bin_counts > 1
        var_of_mean = np.zeros(n_bins, dtype=float)

        if not np.any(valid_bins):
            return bin_means, None

        # Ensure indices are within bounds before indexing
        if not np.all((bin_indices >= 0) & (bin_indices < len(bin_means))):
            raise ValueError("bin_indices contains out-of-bounds values.")

        # Squared deviations from the respective bin mean
        sample_means = bin_means[bin_indices]
        squared_deviations = (predictions - sample_means) ** 2

        # Aggregate squared deviations per bin
        sum_sq_dev = np.bincount(
            bin_indices, weights=squared_deviations, minlength=n_bins
        )

        # Variance of the mean
        var_of_mean[valid_bins] = sum_sq_dev[valid_bins] / (
            bin_counts[valid_bins] * (bin_counts[valid_bins] - 1)
        )

        # Z-score for two-tailed confidence interval
        z_score = stats.norm.ppf(0.5 + confidence_level / 2.0)
        bin_vars = z_score * np.sqrt(var_of_mean)

    return bin_means, bin_vars


def compute_mplot_1d_continuous(
    estimator,
    X,
    feature_idx,
    grid_resolution="auto",
    percentiles=(5, 95),
    confidence_interval=True,
    confidence_level=0.95,
):
    X = np.asarray(X)
    x = X[:, feature_idx]

    quantiles = _build_quantile_grid_1d(
        x, grid_resolution=grid_resolution, percentiles=percentiles
    )
    n_bins = len(quantiles) - 1

    if n_bins < 1:
        raise ValueError(
            f"Feature {feature_idx} has fewer than 2 unique quantile edges. Increase grid_resolution or check your data."
        )

    bin_idx = _bin_indices(x, quantiles)
    predictions = _predict_fn(estimator, X)

    bin_means, bin_vars = _compute_bin_mean_and_variance(
        predictions=predictions,
        n_bins=n_bins,
        bin_indices=bin_idx,
        confidence_interval=confidence_interval,
        confidence_level=confidence_level,
    )

    return bin_means, quantiles, bin_vars


def compute_mplot_1d_categorical(
    estimator,
    X,
    feature_idx,
    percentiles=(5, 95),
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
        low_bnd = np.percentile(x, percentiles[0])
        high_bnd = np.percentile(x, percentiles[1])

        valid_mask = (x >= low_bnd) & (x <= high_bnd)
        X_filtered = X[valid_mask]
        x_filtered = x[valid_mask]
    else:
        X_filtered = X
        x_filtered = x

    unique_values, value_idx = np.unique(x_filtered, return_inverse=True)
    n_values = len(unique_values)

    if n_values < 1:
        raise ValueError(
            f"Feature {feature_idx} has no unique values after filtering. "
            "Check your data or widen your percentiles."
        )

    predictions = _predict_fn(estimator, X_filtered)

    bin_means, bin_vars = _compute_bin_mean_and_variance(
        predictions=predictions,
        n_bins=n_values,
        bin_indices=value_idx,
        confidence_interval=confidence_interval,
        confidence_level=confidence_level,
    )

    return bin_means, unique_values, bin_vars


def compute_mplot_2d(
    estimator,
    X,
    feature_indices,
    grid_resolution="auto",
    percentiles=(5, 95),
):
    feature_indices = list(feature_indices)
    if len(feature_indices) != 2:
        raise ValueError(
            "feature_indices must contain exactly two feature indices."
        )

    X = np.asarray(X)
    idx_i, idx_j = feature_indices
    x_i, x_j = X[:, idx_i], X[:, idx_j]

    quantiles_i = _build_quantile_grid_1d(
        x_i, grid_resolution=grid_resolution, percentiles=percentiles
    )
    quantiles_j = _build_quantile_grid_1d(
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

    # bin_counts = bin_counts_flat.reshape(n_bins_i, n_bins_j)
    bin_means = bin_means_flat.reshape(n_bins_i, n_bins_j)

    return bin_means, quantiles_i, quantiles_j


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
        percentiles=(5, 95),
        confidence_interval=True,
        confidence_level=0.95,
        cmap="viridis",
        **kwargs,
    ):
        if not isinstance(features, (int, np.integer, list)):
            raise ValueError("'features' must be an int or a list of ints.")

        check_is_fitted(self.estimator)

        X = np.asarray(X)

        if isinstance(features, (int, np.integer)):
            feature_type = _check_data_type(
                data_type=feature_type,
                y=X[:, features],
                categorical_max_cardinality=10,
            )

            if self.feature_names is not None:
                feature_name = self.feature_names[features]
            else:
                feature_name = f"X_{features}"

            if feature_type == "continuous":
                bin_means, quantiles, bin_vars = compute_mplot_1d_continuous(
                    self.estimator,
                    X,
                    feature_idx=features,
                    grid_resolution=grid_resolution,
                    percentiles=percentiles,
                    confidence_interval=confidence_interval,
                    confidence_level=confidence_level,
                )

                return self._plot_1d_continuous(
                    X[:, features],
                    bin_means=bin_means,
                    bin_vars=bin_vars,
                    quantiles=quantiles,
                    feature_name=feature_name,
                    cmap=cmap,
                    **kwargs,
                )
            else:
                bin_means, unique_values, bin_vars = (
                    compute_mplot_1d_categorical(
                        self.estimator,
                        X,
                        feature_idx=features,
                        percentiles=percentiles,
                        confidence_interval=confidence_interval,
                        confidence_level=confidence_level,
                    )
                )

                return self._plot_1d_discrete(
                    X[:, features],
                    bin_means=bin_means,
                    bin_vars=bin_vars,
                    uniques_values=unique_values,
                    feature_name=feature_name,
                    cmap=cmap,
                    **kwargs,
                )
        elif isinstance(features, list) and all(
            isinstance(f, (int, np.integer)) for f in features
        ):
            if len(features) > 2:
                raise ValueError(
                    "Only 1D (single int) and 2D (list of two ints) ALE plots are supported."
                )

            if self.feature_names is not None:
                feature_names = [self.feature_names[idx] for idx in features]
            else:
                feature_names = [f"X{idx}" for idx in features]

            bin_means, quantiles_i, quantiles_j = compute_mplot_2d(
                self.estimator,
                X,
                percentiles=percentiles,
                feature_indices=features,
                grid_resolution=grid_resolution,
            )
            return self._plot_2d(
                features_x=X[:, features[0]],
                features_y=X[:, features[1]],
                bin_means=bin_means,
                quantiles_i=quantiles_i,
                quantiles_j=quantiles_j,
                feature_names=feature_names,
                cmap=cmap,
                **kwargs,
            )
        else:
            raise TypeError("'features' must be an int or a list of int.")

    @staticmethod
    def _plot_1d_continuous(
        feature_values,
        bin_means,
        bin_vars,
        quantiles,
        feature_name,
        cmap=None,
        **kwargs,
    ):
        """Render a 1D continuous M-plot with a marginal density strip."""
        del cmap

        centers = (quantiles[1:] + quantiles[:-1]) / 2
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
            n_levels=20,
        )
        sns.despine(ax=ax_top, left=True)
        ax_top.tick_params(axis="x", which="both", bottom=False, top=False)
        ax_top.yaxis.set_visible(False)

        ax_main = axes[1]
        sns.lineplot(x=centers, y=bin_means, ax=ax_main, **kwargs)
        if bin_vars is not None:
            ax_main.fill_between(
                centers,
                bin_means - bin_vars,
                bin_means + bin_vars,
                color="b",
                alpha=0.15,
                label="Confidence Interval",
            )
        ax_main.set_xlim(low - margin, high + margin)
        ax_main.set_xlabel(feature_name)
        ax_main.set_ylabel("M-plot")

        sns.despine(ax=ax_main)
        plt.tight_layout()
        return axes

    @staticmethod
    def _plot_1d_discrete(
        feature_values,
        bin_means,
        bin_vars,
        uniques_values,
        feature_name,
        cmap=None,
        **kwargs,
    ):
        """Render a 1D discrete M-plot with a marginal density histogram."""
        del cmap

        low, high = (
            uniques_values.min(),
            uniques_values.max(),
        )

        if uniques_values.dtype.kind in "iuf":
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
            x=uniques_values,
            y=bin_means,
            ax=ax_main,
            marker="o",
            **kwargs,
        )
        if bin_vars is not None:
            ax_main.fill_between(
                uniques_values,
                bin_means - bin_vars,
                bin_means + bin_vars,
                color="b",
                alpha=0.15,
            )
        ax_main.xaxis.set_ticks(uniques_values)
        ax_main.set_xlim(low - 0.6, high + 0.6)
        ax_main.set_xlabel(feature_name)
        ax_main.set_ylabel("M-plot")

        sns.despine(ax=ax_main)
        plt.tight_layout()
        return axes

    @staticmethod
    def _plot_2d(
        features_x,
        features_y,
        bin_means,
        quantiles_i,
        quantiles_j,
        feature_names,
        cmap="viridis",
        **kwargs,
    ):
        """Render a 2D M-plot with marginal density strips on each axis."""
        low_i, high_i = quantiles_i.min(), quantiles_i.max()
        low_j, high_j = quantiles_j.min(), quantiles_j.max()

        """
        This is what was plotted instead of bin_means.T with ax_main.pcolormesh()
        zz_cells = np.empty(
            (bin_means.shape[0] - 1, bin_means.shape[1] - 1),
            dtype=bin_means.dtype,
        )
        zz_cells[:] = (
            bin_means[:-1, :-1]
            + bin_means[1:, 1:]
            + bin_means[:-1, 1:]
            + bin_means[1:, :-1]
        ) / 4"""

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
            bin_means.T,
            cmap=cmap,
            shading="flat",
            edgecolors="face",
            **kwargs,
        )

        level_lines = [0.1, 0.3, 0.5, 0.7, 0.9]
        # Downsample to avoid running OOM for large N (kde is O(N^2))
        n_samples_kde = min(len(features_x), 5000)
        idx = np.random.default_rng(42).permutation(len(features_x))[
            :n_samples_kde
        ]

        sns.kdeplot(
            x=features_x[idx],
            y=features_y[idx],
            ax=ax_main,
            levels=level_lines,
            colors="black",
            linewidths=0.8,
            alpha=0.7,
        )
        cs = ax_main.collections[-1]

        contour_labels = {lvl: f"{lvl:.1f}" for lvl in level_lines}
        cs = ax_main.collections[-1]
        ax_main.clabel(
            cs,
            inline=True,
            fontsize=9,
            colors="black",
            fmt=contour_labels,
        )

        ax_main.set_xlabel(feature_names[0])
        ax_main.set_ylabel(feature_names[1])
        ax_main.set_xlim(low_i, high_i)
        ax_main.set_ylim(low_j, high_j)

        ax_top = axes[0, 0]
        sns.kdeplot(
            features_x[idx],
            ax=ax_top,
            fill=True,
            color="black",
            alpha=0.25,
            legend=False,
        )
        sns.despine(ax=ax_top, left=True)
        ax_top.xaxis.set_ticks([])
        ax_top.yaxis.set_visible(False)
        ax_top.set_xlim(low_i, high_i)

        ax_right = axes[1, 2]
        sns.kdeplot(
            y=features_y[idx],
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
