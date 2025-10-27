from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.inspection import partial_dependence


class PDP:
    def __init__(self, estimator, feature_names=None):
        self.estimator = estimator
        self.feature_names = feature_names

    def plot(self, X, features, cmap="viridis", **kwargs):
        if isinstance(features, int):
            feature_ids = [features]
            plotting_func = self._plot_1d
        elif isinstance(features, list):
            if len(features) > 2:
                raise ValueError("Only 1D and 2D PDP plots are supported")
            else:
                feature_ids = copy(features)
                plotting_func = self._plot_2d

        if self.feature_names is not None:
            feature_names = [self.feature_names[idx] for idx in feature_ids]
        else:
            feature_names = [f"X{idx}" for idx in feature_ids]

        print(features, feature_ids, feature_names)
        pd = partial_dependence(self.estimator, X, features=features)
        return plotting_func(pd, feature_names, cmap=cmap, **kwargs)

    @staticmethod
    def _plot_1d(pd, feature_names, cmap=None, **kwargs):

        _, axes = plt.subplots(2, 1, height_ratios=[0.2, 1])
        ax = axes[0]

        sns.kdeplot(pd["grid_values"], ax=ax, legend=False, fill=True)
        sns.rugplot(pd["grid_values"], ax=ax, height=0.25, legend=False)
        sns.despine(ax=ax, left=True)
        # Plot partial dependence
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_visible(False)

        ax = axes[1]
        sns.lineplot(x=pd["grid_values"][0], y=pd["average"][0], **kwargs)
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel("Partial Dependence")
        sns.despine(ax=ax)
        plt.tight_layout()
        return axes

    @staticmethod
    def _plot_2d(pd, feature_names, cmap="viridis", **kwargs):
        x = pd["grid_values"][0]
        y = pd["grid_values"][1]
        z = pd["average"][0]

        xx, yy = np.meshgrid(x, y, indexing="ij")

        _, axes = plt.subplots(
            2, 2, figsize=(8, 6), height_ratios=[0.2, 1], width_ratios=[1, 0.2]
        )
        ax = axes[1, 0]
        contour = ax.contour(xx, yy, z, cmap=cmap, **kwargs)
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.clabel(contour, inline=True, fontsize=10)
        sns.despine(ax=ax)

        ax = axes[0, 0]
        sns.kdeplot(x, ax=ax, legend=False, fill=True)
        sns.rugplot(x, ax=ax, height=0.25, legend=False)
        sns.despine(ax=ax)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_visible(False)

        ax = axes[1, 1]
        sns.kdeplot(y=y, ax=ax, legend=False, fill=True)
        sns.rugplot(y=y, ax=ax, height=0.25, legend=False)
        sns.despine(ax=ax)
        ax.spines["bottom"].set_visible(False)
        ax.yaxis.set_ticks([])
        ax.xaxis.set_visible(False)

        axes[0, 1].remove()
        plt.tight_layout()
        return axes
