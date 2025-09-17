import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_importance(
    importances,
    feature_names=None,
    ax=None,
    ascending=False,
    **kwargs,
):
    """
    Plot feature importances as a horizontal bar plot.

    Parameters
    ----------
    importances : array-like of shape (n_features,) or (n_repeats, n_features)
        Feature importance scores. If 2D, each column represents a different repetition.
    feature_names : list of str
        Names of the features.
    ax : matplotlib.axes.Axes, optional (default=None)
        Axes object to draw the plot onto, otherwise uses the current Axes.
    ascending: bool, optional (default=False)
        Whether to sort features by ascending importance.
    **kwargs : additional keyword arguments
        Additional arguments passed to seaborn.barplot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object with the plot.
    """
    if ax is None:
        _, ax = plt.subplots()
    if feature_names is None:
        feature_names = [f"x_{i}" for i in range(len(importances))]

    if importances.ndim == 2:
        df_plot = {
            "Feature": feature_names * importances.shape[0],
            "Importance": importances.flatten(),
        }
    else:
        df_plot = {"Feature": feature_names, "Importance": importances}

    df_plot = pd.DataFrame(df_plot)

    # Sort features by decreasing mean importance
    mean_importance = df_plot.groupby("Feature")["Importance"].mean()
    sorted_features = mean_importance.sort_values(ascending=ascending).index
    df_plot["Feature"] = pd.Categorical(
        df_plot["Feature"],
        categories=sorted_features,
        ordered=True,
    )
    sns.barplot(df_plot, x="Importance", y="Feature", ax=ax, **kwargs)
    sns.despine(ax=ax)
    ax.set_ylabel("")
    return ax
