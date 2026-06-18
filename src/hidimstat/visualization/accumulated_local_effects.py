from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ._compute_accumulated_local_effects import (
    compute_ale_1d_continuous,
    compute_ale_1d_discrete,
    compute_ale_2d,
)


class ALE:
    r"""
    Accumulated Local Effect (ALE) visualisation.

    ALE measures how the predictions of a model change on average when a
    feature varies locally within small intervals (bins).  Unlike Partial
    Dependence Plots, ALE is not affected by feature correlations because it
    averages *local* differences rather than marginalising over the full
    feature distribution.

    Formally, for a single continuous feature :math:`x_j`, the 1D ALE is:

    .. math::

        \hat{f}_{j,\text{ALE}}(x) =
            \int_{z_{0,j}}^{x}
            \mathbb{E}\!\left[
                \frac{\partial f(X)}{\partial X_j}
                \;\middle|\; X_j = z
            \right] dz - c

    where :math:`c` is a centering constant chosen so that the ALE averages
    to zero over the training distribution.

    Parameters
    ----------
    estimator : fitted sklearn-compatible estimator
        Must expose ``predict``, ``predict_proba``, or ``decision_function``.
    feature_names : list of str, optional
        Human-readable names for each column of ``X``.  If ``None``, labels
        default to ``X0``, ``X1``, ...
    """

    def __init__(self, estimator, feature_names=None):
        self.estimator = estimator
        self.feature_names = feature_names

    def _resolve_feature_type(self, X, feature, feature_type):
        """Recognize the feature type to use when feature_type="auto" in plot()"""
        if feature_type == "auto":
            if X[:, feature].dtype.kind in "iuf":
                len_unique_values = len(np.unique(X[:, feature]))
                if (
                    len_unique_values <= 10
                    or len_unique_values / len(X) <= 0.001
                ):
                    return "discrete"
                return "continuous"
            return "categorical"

        if feature_type in ["continuous", "discrete", "categorical"]:
            return feature_type

        raise ValueError(
            "feature_type should be a string among 'auto', 'discrete', 'continuous', or 'categorical'"
        )

    def plot(
        self,
        X,
        features,
        feature_type="auto",
        grid_resolution=50,
        cmap="viridis",
        **kwargs,
    ):
        """Compute and display the ALE plot for one or two features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dataset used to build the quantile grid and to gather local effects.
        features : int or list of int
            Feature index (1D ALE) or pair of feature indices (2D ALE).
        feature_type : string among 'auto', 'discrete', 'continuous', or 'categorical'
            Specify the type of values the feature has for 1D ALE. Set by default to auto aand in this case :
            - non-numeric feature : categorical
            - numeric feature : discrete if the feature has less than 10 unique values or the number of unique values
            is less than 0.1% of the samples, and continuous otherwise
        grid_resolution : int, default=50
            Number of bins per feature axis.
        cmap : str, default="viridis"
            Matplotlib colormap used for the 2D contour plot.
        **kwargs
            Extra keyword arguments forwarded to:

            - ``sns.lineplot`` for 1D plots;
            - ``ax.pcolormesh`` for 2D plots.

        Returns
        -------
        axes : ndarray of Axes
            The matplotlib ``Axes`` objects that make up the figure.
        """
        X = np.asarray(X)

        if isinstance(features, int):
            feature_ids = [features]
            feature_type = self._resolve_feature_type(
                X, features, feature_type
            )
            if feature_type == "continuous":
                plotting_func = self._plot_1d_continuous
                result = compute_ale_1d_continuous(
                    self.estimator,
                    X,
                    feature_idx=features,
                    grid_resolution=grid_resolution,
                )
            elif feature_type == "discrete":
                plotting_func = self._plot_1d_discrete
                result = compute_ale_1d_discrete(
                    self.estimator, X, feature_idx=features
                )
            else:
                raise ValueError("Categorical not yet implemented.")
        elif isinstance(features, list):
            if len(features) > 2:
                raise ValueError(
                    "Only 1D (single int) and 2D (list of two ints) ALE plots are supported."
                )
            feature_ids = copy(features)
            plotting_func = self._plot_2d
            result = compute_ale_2d(
                self.estimator,
                X,
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
            result, X, feature_ids, feature_names, cmap=cmap, **kwargs
        )

    @staticmethod
    def _plot_1d_continuous(
        result, X, feature_ids, feature_names, cmap=None, **kwargs
    ):
        """Render a 1D continuous ALE plot with a marginal density strip."""
        del cmap  # only there for API compatibility

        feature_values = X[:, feature_ids[0]]

        _, axes = plt.subplots(2, 1, height_ratios=[0.2, 1], sharex=True)

        # Top strip: marginal distribution of the feature
        ax_top = axes[0]
        sns.kdeplot(feature_values, ax=ax_top, fill=True, legend=False)
        sns.despine(ax=ax_top, left=True)
        ax_top.tick_params(axis="x", which="both", bottom=False, top=False)
        ax_top.yaxis.set_visible(False)

        # Main panel: ALE curve
        ax_main = axes[1]
        sns.lineplot(
            x=result["quantiles"], y=result["ale"], ax=ax_main, **kwargs
        )
        ax_main.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax_main.set_xlabel(feature_names[0])
        ax_main.set_ylabel("ALE")
        sns.despine(ax=ax_main)
        plt.tight_layout()
        return axes

    @staticmethod
    def _plot_1d_discrete(
        result, X, feature_ids, feature_names, cmap=None, **kwargs
    ):
        """Render a 1D discrete ALE plot with a marginal density histogram."""
        del cmap  # only there for API compatibility

        feature_values = X[:, feature_ids[0]]

        _, axes = plt.subplots(
            2, 1, figsize=(8, 4), height_ratios=[0.2, 1], sharex=True
        )

        # Top strip: marginal distribution of the feature
        ax_top = axes[0]
        sns.histplot(
            feature_values, ax=ax_top, discrete=True, fill=True, legend=False
        )
        sns.despine(ax=ax_top, left=True)
        ax_top.tick_params(axis="x", which="both", bottom=False, top=False)
        ax_top.yaxis.set_visible(False)

        # Main panel: ALE curve
        ax_main = axes[1]
        sns.lineplot(
            x=result["unique_values"],
            y=result["ale"],
            ax=ax_main,
            marker="o",
            **kwargs,
        )
        ax_main.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax_main.xaxis.set_ticks(result["unique_values"])
        ax_main.set_xlabel(feature_names[0])
        ax_main.set_ylabel("ALE")
        sns.despine(ax=ax_main)
        plt.tight_layout()
        return axes

    @staticmethod
    def _plot_2d(
        result, X, feature_ids, feature_names, cmap="viridis", **kwargs
    ):
        """Render a 2D ALE plot with marginal density strips on each axis."""
        x = X[:, feature_ids[0]]
        y = X[:, feature_ids[1]]

        quantiles_i = result["quantiles_i"]
        quantiles_j = result["quantiles_j"]
        ale = result["ale"]

        zz_cells = (
            ale[:-1, :-1] + ale[1:, 1:] + ale[:-1, 1:] + ale[1:, :-1]
        ) / 4.0

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(8, 6),
            height_ratios=[0.2, 1],
            width_ratios=[1, 0.2, 0.05],
        )

        axes[0, 1].axis("off")
        axes[0, 2].axis("off")

        # Main panel: ALE map
        ax_main = axes[1, 0]
        mesh = ax_main.pcolormesh(
            quantiles_i,
            quantiles_j,
            zz_cells.T,
            cmap=cmap,
            shading="flat",
            edgecolors="face",
            **kwargs,
        )
        ax_main.set_xlabel(feature_names[0])
        ax_main.set_ylabel(feature_names[1])
        ax_main.set_xlim(quantiles_i.min(), quantiles_i.max())
        ax_main.set_ylim(quantiles_j.min(), quantiles_j.max())

        # Top strip: marginal of feature 0
        ax_top = axes[0, 0]
        sns.kdeplot(x, ax=ax_top, fill=True, legend=False)
        sns.despine(ax=ax_top, left=True)
        ax_top.xaxis.set_ticks([])
        ax_top.yaxis.set_visible(False)

        # Right strip: marginal of feature 1
        ax_right = axes[1, 1]
        sns.kdeplot(y=y, ax=ax_right, fill=True, legend=False)
        sns.despine(ax=ax_right, bottom=True)
        ax_right.yaxis.set_ticks([])
        ax_right.xaxis.set_visible(False)

        # Color bar
        ax_cbar = axes[1, 2]
        cbar = fig.colorbar(mesh, cax=ax_cbar)
        cbar.set_label("ALE", rotation=270, labelpad=15)

        plt.tight_layout()
        return axes
