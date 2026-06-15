from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ._compute_accumulated_local_effects import compute_ale_1d, compute_ale_2d


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


    def plot(self, X, features, grid_resolution=50, is_categorical=False, cmap="viridis", **kwargs):
        """Compute and display the ALE plot for one or two features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dataset used to build the quantile grid and to gather local effects.
        features : int or list of int
            Feature index (1D ALE) or pair of feature indices (2D ALE).
        grid_resolution : int, default=50
            Number of bins per feature axis.
        is_categorical : bool or list of two bools, default=False
            Placeholder for future categorical support.  Must be ``False``
            (or ``[False, False]`` for 2D).
        cmap : str, default="viridis"
            Matplotlib colormap used for the 2D contour plot.
        **kwargs
            Extra keyword arguments forwarded to:

            - ``sns.lineplot`` for 1D plots;
            - ``ax.contourf`` for 2D plots.

        Returns
        -------
        axes : ndarray of Axes
            The matplotlib ``Axes`` objects that make up the figure.
        """
        X = np.asarray(X, dtype=float)

        if isinstance(features, int):
            feature_ids = [features]
            is_cat_arg = bool(is_categorical)
            plotting_func = self._plot_1d
        elif isinstance(features, list):
            if len(features) > 2:
                raise ValueError("Only 1D (single int) and 2D (list of two ints) ALE plots are supported.")
            feature_ids = copy(features)
            plotting_func = self._plot_2d
        else:
            raise TypeError("'features' must be an int or a list of int.")

        if self.feature_names is not None:
            feature_names = [self.feature_names[idx] for idx in feature_ids]
        else:
            feature_names = [f"X{idx}" for idx in feature_ids]

        # Compute
        if isinstance(features, int):
            result = compute_ale_1d(self.estimator, X, feature_idx=features, grid_resolution=grid_resolution)
        else:
            result = compute_ale_2d(self.estimator, X, feature_indices=features, grid_resolution=grid_resolution)

        return plotting_func(result, X, feature_ids, feature_names, cmap=cmap, **kwargs)

    @staticmethod
    def _plot_1d(result, X, feature_ids, feature_names, cmap=None, **kwargs):
        """Render a 1D ALE plot with a marginal density strip."""
        del cmap  # only present for API symmetry with _plot_2d

        feature_values = X[:, feature_ids[0]]

        _, axes = plt.subplots(2, 1, height_ratios=[0.2, 1])

        # Top strip: marginal distribution of the feature
        ax_top = axes[0]
        sns.kdeplot(feature_values, ax=ax_top, fill=True, legend=False)
        sns.rugplot(feature_values, ax=ax_top, height=0.25, legend=False)
        sns.despine(ax=ax_top, left=True)
        ax_top.spines["left"].set_visible(False)
        ax_top.spines["bottom"].set_visible(True)
        ax_top.xaxis.set_ticks([])
        ax_top.yaxis.set_visible(False)

        # Main panel: ALE curve
        ax_main = axes[1]
        sns.lineplot(x=result["quantiles"], y=result["uncentered_ale"], ax=ax_main, **kwargs)
        ax_main.axhline(result["uncentered_mean"], color="grey", linewidth=0.8, linestyle="--")
        ax_main.set_xlabel(feature_names[0])
        ax_main.set_ylabel("ALE")
        sns.despine(ax=ax_main)
        plt.tight_layout()
        return axes

    @staticmethod
    def _plot_2d(result, X, feature_ids, feature_names, cmap="viridis", **kwargs):
        """Render a 2D ALE plot with marginal density strips on each axis."""
        x0_vals = X[:, feature_ids[0]]
        x1_vals = X[:, feature_ids[1]]

        xx, yy = np.meshgrid(result["quantiles_i"], result["quantiles_j"], indexing="ij")
        zz = result["ale"]

        _, axes = plt.subplots(
            2, 2, figsize=(8, 6), height_ratios=[0.2, 1], width_ratios=[1, 0.2]
        )

        # Main panel: filled contour
        ax_main = axes[1, 0]
        contour = ax_main.contour(xx, yy, zz, cmap=cmap, **kwargs)
        ax_main.set_xlabel(feature_names[0])
        ax_main.set_ylabel(feature_names[1])
        ax_main.clabel(contour, inline=True, fontsize=10)
        sns.despine(ax=ax_main)

        # Top strip: marginal of feature 0
        ax_top = axes[0, 0]
        sns.kdeplot(x0_vals, ax=ax_top, fill=True, legend=False)
        sns.rugplot(x0_vals, ax=ax_top, height=0.25, legend=False)
        sns.despine(ax=ax_top, left=True)
        ax_top.spines["left"].set_visible(False)
        ax_top.spines["bottom"].set_visible(True)
        ax_top.xaxis.set_ticks([])
        ax_top.yaxis.set_visible(False)

        # Right strip: marginal of feature 1
        ax_right = axes[1, 1]
        sns.kdeplot(y=x1_vals, ax=ax_right, fill=True, legend=False)
        sns.rugplot(y=x1_vals, ax=ax_right, height=0.25, legend=False)
        sns.despine(ax=ax_right, bottom=True)
        ax_right.spines["bottom"].set_visible(False)
        ax_right.yaxis.set_ticks([])
        ax_right.xaxis.set_visible(False)

        axes[0, 1].remove()
        plt.tight_layout()
        return axes
