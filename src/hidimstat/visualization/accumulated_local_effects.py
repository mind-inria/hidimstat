from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def _predict_fn(estimator, X):
    """Return a scalar-output prediction callable for *estimator*.

    For classifiers we use `predict_proba` (returns the probability of the
    positive class for binary problems) when available, otherwise
    `decision_function`. For regressors we use `predict`.

    Parameters
    ----------
    estimator : fitted sklearn-compatible estimator
    X : array-like of shape (n_samples, n_features)

    Returns
    -------
    pred : ndarray of shape (n_samples,)
        A 1D array containing the scalar-output predictions.
    """
    feature_names = getattr(estimator, "feature_names_in_", None)

    if hasattr(estimator, "predict_proba"):
        prediction_function = "predict_proba"
    elif hasattr(estimator, "decision_function"):
        prediction_function = "decision_function"
    elif hasattr(estimator, "predict"):
        prediction_function = "predict"
    else:
        raise ValueError(
            "'estimator' must expose at least one of predict_proba, decision_function, or predict."
        )

    if feature_names is not None:
        X = pd.DataFrame(X, columns=feature_names)

    pred = getattr(estimator, prediction_function)(X)

    if (hasattr(estimator, "classes_") and len(estimator.classes_) > 2) or (
        pred.ndim == 2 and pred.shape[1] > 2
    ):
        raise ValueError("Multiclass models are not supported.")

    # Binary: keep only the positive-class column
    if (
        prediction_function == "predict_proba"
        and pred.ndim == 2
        and pred.shape[1] == 2
    ):
        return pred[:, 1]

    return pred.ravel()


def _build_quantile_grid(x, grid_resolution, percentiles=(0.05, 0.95)):
    """Build a 1D quantile grid for a single continuous feature.

    Parameters
    ----------
    x : ndarray of shape (n_samples,)
        Values for one feature.
    grid_resolution : int or "auto", default="auto"
        Number of bins in the grid. Set by default to "auto".

        - If "auto", the number of bins is determined automatically
          to minimize the histogram error.
        - Note that the final number of bins in the returned grid may be
          strictly less than `grid_resolution` (or the auto-calculated value)
          if the data contains many duplicate values or fewer unique points
          than requested.

    percentiles : tuple of float, default=(0.05, 0.95)
        The lower and upper percentile used to create the extreme values for the grid.
        Must be in [0, 1].


    Returns
    -------
    quantiles : ndarray of shape (n_quantiles,)
        Unique, sorted quantile bins edges.
    """
    if (
        not isinstance(percentiles, tuple)
        or len(percentiles) != 2
        or not (0.0 <= percentiles[0] <= percentiles[1] <= 1.0)
    ):
        raise ValueError(
            "'percentiles' must be a tuple of 2 floats "
            "in [0, 1] in increasing order"
        )

    low_bnd = np.percentile(x, percentiles[0] * 100)
    high_bnd = np.percentile(x, percentiles[1] * 100)

    valid_mask = (x >= low_bnd) & (x <= high_bnd)
    x_filtered = x[valid_mask]

    if grid_resolution == "auto":
        grid_resolution = (
            np.histogram_bin_edges(x_filtered, bins="auto").size - 1
        )

    if (
        not isinstance(grid_resolution, (int, np.integer))
        or grid_resolution <= 0
    ):
        raise ValueError(
            "'grid_resolution' must be an int strictly greater than 0 or 'auto'."
        )

    # Use unique values when there are fewer than grid_resolution unique points
    uniques = np.unique(x_filtered)
    if uniques.shape[0] <= grid_resolution:
        return uniques

    probs = np.linspace(0.0, 1.0, grid_resolution + 1)
    return np.unique(
        np.percentile(x_filtered, probs * 100, method="inverted_cdf")
    )


def _bin_indices(x, quantiles):
    """Assign each sample to a bin defined by *quantiles*.

    Samples are placed in bin `k` when `quantiles[k] <= x < quantiles[k+1]`.
    The last bin is closed on the right. Indices are clipped so that every
    sample falls within `[0, len(quantiles) - 2]`.

    Parameters
    ----------
    x : ndarray of shape (n_samples,)
    quantiles : ndarray of shape (n_quantiles,)

    Returns
    -------
    indices : ndarray of shape (n_samples,), dtype int
    """
    # digitize returns 0 for x < quantiles[0] and len(quantiles) for x > quantiles[-1]
    idx = np.digitize(x, quantiles) - 1
    # samples equal to the last edge fall in the last bin
    return np.clip(idx, 0, len(quantiles) - 2)


def compute_ale_1d_continuous(
    estimator,
    X,
    feature_idx,
    grid_resolution="auto",
    confidence_interval=False,
    confidence_level=0.95,
    percentiles=(0.05, 0.95),
):
    """Compute the 1D Accumulated Local Effect for a single continuous feature.

    For each bin defined by the quantile grid of `X[:, feature_idx]`,
    the local effect is estimated as the average difference in model output
    when the feature moves from the lower to the upper bin edge across all
    samples that fall within that bin. The cumulative sum of these average
    effects gives the (uncentered) ALE curve, which is then centered so its
    weighted mean is zero.

    Parameters
    ----------
    estimator : fitted sklearn-compatible estimator
        Must expose `predict`, `predict_proba`, or `decision_function`.
    X : array-like of shape (n_samples, n_features)
        Training (or evaluation) dataset used to define the quantile grid and
        to gather local samples in each bin.
    feature_idx : int
        Column index of the feature of interest.
    grid_resolution : int or "auto", default="auto"
        Number of bins used to build the quantile grid. Set by default to "auto".

        - If "auto", the number of bins is determined automatically
          to minimize the histogram error.
        - Note that the final number of bins in the quantile grid may be
          strictly less than `grid_resolution` (or the auto-calculated value)
          if the data contains many duplicate values or fewer unique points
          than requested.

    confidence_interval : bool, default=False
        Whether to compute the confidence intervals of the ALE curve.
    confidence_level : float, default=0.95
        The confidence level used to compute the confidence intervals (e.g., 0.95 for 95%).
    percentiles : tuple of float, default=(0.05, 0.95)
        The lower and upper percentile used to create the extreme values for the grid.
        Must be in [0, 1].

    Returns
    -------
    result : dict with the following keys:

        `"ale"` : ndarray of shape (n_quantiles,)
            ALE values evaluated at each bin edge.
        `"quantiles"` : ndarray of shape (n_quantiles,)
            Bin edges (quantiles of the feature distribution).
        `"ale_err"` : ndarray of shape (n_quantiles,) or None
            The margin of error for each quantile boundary at the specified confidence level.
            Returns `None` if `confidence_interval` is False.
    """
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

    # For each sample, evaluate the model at the lower and upper edge of its bin
    X_low = X.copy()
    X_high = X.copy()
    X_low[:, feature_idx] = quantiles[bin_idx]
    X_high[:, feature_idx] = quantiles[bin_idx + 1]

    local_effects = _predict_fn(estimator, X_high) - _predict_fn(
        estimator, X_low
    )  # shape (n_samples,)

    # Average local effects within each bin
    bin_counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
    bin_sums = np.bincount(bin_idx, weights=local_effects, minlength=n_bins)

    mean_effects = np.zeros(n_bins, dtype=float)
    non_zero = bin_counts > 0
    mean_effects[non_zero] = bin_sums[non_zero] / bin_counts[non_zero]

    # Cumulative sum: uncentered ALE evaluated for each bin edge
    ale = np.array([0, *np.cumsum(mean_effects)])

    # Center: subtract the sample-weighted mean
    ale_centers = (ale[1:] + ale[:-1]) / 2
    ale -= np.sum(ale_centers * bin_counts) / bin_counts.sum()

    # Confidence interval
    ale_err = None
    if confidence_interval:
        sample_means = mean_effects[bin_idx]
        squared_deviations = (local_effects - sample_means) ** 2
        sum_squared_deviations = np.bincount(
            bin_idx, weights=squared_deviations, minlength=n_bins
        )

        var_of_mean = np.zeros(n_bins, dtype=float)
        valid_bins = bin_counts > 1
        var_of_mean[valid_bins] = sum_squared_deviations[valid_bins] / (
            bin_counts[valid_bins] * (bin_counts[valid_bins] - 1)
        )

        var_of_mean = np.array([0, *var_of_mean])
        z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        ale_err = z_score * np.sqrt(var_of_mean)

    return {
        "ale": ale,
        "quantiles": quantiles,
        "ale_err": ale_err,
    }


def compute_ale_1d_discrete(
    estimator,
    X,
    feature_idx,
    confidence_interval=False,
    confidence_level=0.95,
    percentiles=(0.05, 0.95),
):
    """Compute the 1D Accumulated Local Effect for a single discrete feature.

    For each value of the feature, the local effect is estimated as the average
    difference in model output when the feature moves from the lower to the upper
    neighbour values across all samples that have that value. The cumulative sum
    of these average effects gives the (uncentered) ALE curve, which is then
    centered so its weighted mean is zero.

    Parameters
    ----------
    estimator : fitted sklearn-compatible estimator
        Must expose `predict`, `predict_proba`, or `decision_function`.
    X : array-like of shape (n_samples, n_features)
        Training (or evaluation) dataset used to define the quantile grid and
        to gather local samples in each bin.
    feature_idx : int
        Column index of the feature of interest.
    confidence_interval : bool, default=False
        Whether to compute the confidence intervals of the ALE curve.
    confidence_level : float, default=0.95
        The confidence level used to compute the confidence intervals (e.g., 0.95 for 95%).
    percentiles : tuple of float, default=(0.05, 0.95)
        The lower and upper percentile used to create the extreme values for the grid.
        Must be in [0, 1].

    Returns
    -------
    result : dict with the following keys:

        `"ale"` : ndarray of shape (n_quantiles,)
            ALE values evaluated at each value of the feature.
        `"unique_values"` : ndarray of shape (n_values,)
            Unique values of the feature of interest.
        `"ale_err"` : ndarray of shape (n_values,) or None
            The margin of error for each discrete value at the specified confidence level.
            Returns `None` if `confidence_interval` is False.
    """
    if (
        not isinstance(percentiles, tuple)
        or len(percentiles) != 2
        or not (0.0 <= percentiles[0] <= percentiles[1] <= 1.0)
    ):
        raise ValueError(
            "'percentiles' must be a tuple of 2 floats "
            "in [0, 1] in increasing order"
        )

    X = np.asarray(X)
    x = X[:, feature_idx]

    low_bnd = np.percentile(x, percentiles[0] * 100)
    high_bnd = np.percentile(x, percentiles[1] * 100)

    valid_mask = (x >= low_bnd) & (x <= high_bnd)
    X_filtered = X[valid_mask]
    x_filtered = x[valid_mask]

    unique_values = np.unique(x_filtered)
    n_values = len(unique_values)

    if n_values < 2:
        raise ValueError(
            f"Feature {feature_idx} has fewer than 2 unique values. Check your data."
        )

    value_idx = np.digitize(x_filtered, unique_values) - 1

    # For each sample, evaluate the model at the lower and upper edge of its bin
    X_low = X_filtered.copy()
    X_high = X_filtered.copy()
    mask_low = x_filtered != unique_values[0]
    mask_high = x_filtered != unique_values[-1]
    X_low[mask_low, feature_idx] = unique_values[value_idx[mask_low] - 1]
    X_high[mask_high, feature_idx] = unique_values[value_idx[mask_high] + 1]

    local_effects_low = _predict_fn(
        estimator, X_filtered[mask_low]
    ) - _predict_fn(estimator, X_low[mask_low])
    local_effects_high = _predict_fn(
        estimator, X_high[mask_high]
    ) - _predict_fn(estimator, X_filtered[mask_high])

    # Average local effects within each bin
    combined_idx = np.concatenate(
        [value_idx[mask_low], value_idx[mask_high] + 1]
    )
    combined_effects = np.concatenate([local_effects_low, local_effects_high])

    value_counts = np.bincount(combined_idx, minlength=n_values).astype(float)
    value_sums = np.bincount(
        combined_idx, weights=combined_effects, minlength=n_values
    )

    mean_effects = np.zeros(n_values, dtype=float)
    non_zero = value_counts > 0
    mean_effects[non_zero] = value_sums[non_zero] / value_counts[non_zero]

    # Cumulative sum: uncentered ALE evaluated for each bin edge
    ale = np.cumsum(mean_effects)

    # Center: subtract the sample-weighted mean
    ale_centers = (ale[1:] + ale[:-1]) / 2
    ale -= np.sum(ale_centers * value_counts[1:]) / value_counts.sum()

    # Confidence interval
    ale_err = None
    if confidence_interval:
        sample_means = mean_effects[combined_idx]
        squared_deviations = (combined_effects - sample_means) ** 2
        sum_squared_deviations = np.bincount(
            combined_idx, weights=squared_deviations, minlength=n_values
        )

        var_of_mean = np.zeros(n_values, dtype=float)
        valid_values = value_counts > 1
        var_of_mean[valid_values] = sum_squared_deviations[valid_values] / (
            value_counts[valid_values] * (value_counts[valid_values] - 1)
        )

        z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        ale_err = z_score * np.sqrt(var_of_mean)

    return {"ale": ale, "unique_values": unique_values, "ale_err": ale_err}


def compute_ale_2d(
    estimator,
    X,
    feature_indices,
    grid_resolution="auto",
    percentiles=(0.05, 0.95),
):
    """Compute the 2D Accumulated Local Effect for a pair of continuous features.

    The 2D ALE isolates the *interaction* effect between two features by
    accumulating second-order local differences across a 2D grid of quantile
    bins. The resulting surface captures the joint effect that cannot be
    explained by either 1D ALE alone.

    Parameters
    ----------
    estimator : fitted sklearn-compatible estimator
    X : array-like of shape (n_samples, n_features)
        Training (or evaluation) dataset used to define the quantile grid and
        to gather local samples in each bin.
    feature_indices : tuple or list of two ints
        Column indices `[i, j]` of the two features of interest.
    grid_resolution : int or "auto", default="auto"
        Number of bins per feature axis. Set by default to "auto".

        - If "auto", the number of bins is determined automatically
          to minimize the histogram error per feature axis.
        - Note that the final number of bins in the quantile grids may be
          strictly less than `grid_resolution` (or the auto-calculated value)
          if the data contains many duplicate values or fewer unique points
          than requested.

    percentiles : tuple of float, default=(0.05, 0.95)
        The lower and upper percentile used to create the extreme values for the grid.
        Must be in [0, 1].


    Returns
    -------
    result : dict with the following keys:

        `"ale"` : ndarray of shape (n_quantiles_i, n_quantiles_j)
            2D ALE values evaluated at each 2D bin corner.
        `"quantiles_i"` : ndarray of shape (n_quantiles_i,)
            Bin edges for the first feature.
        `"quantiles_j"` : ndarray of shape (n_quantiles_j,)
            Bin edges for the second feature.
    """
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
    bin_idx = [bin_idx_i, bin_idx_j]

    # Second-order finite differences: evaluate at the four corners of each 2D bin
    predictions = {}
    X_copy = X.copy()

    for offset_i in range(2):
        for offset_j in range(2):
            X_copy[:, idx_i] = quantiles_i[bin_idx_i + offset_i]
            X_copy[:, idx_j] = quantiles_j[bin_idx_j + offset_j]
            predictions[(offset_i, offset_j)] = _predict_fn(estimator, X_copy)

    # Second-order local effects
    local_effects = (predictions[(1, 1)] - predictions[(1, 0)]) - (
        predictions[(0, 1)] - predictions[(0, 0)]
    )  # shape (n_samples,)

    # Average local effects within each 2D bin
    flat_bin_idx = bin_idx_i * n_bins_j + bin_idx_j
    n_flat_bins = n_bins_i * n_bins_j

    bin_counts_flat = np.bincount(flat_bin_idx, minlength=n_flat_bins).astype(
        float
    )
    bin_sums_flat = np.bincount(
        flat_bin_idx, weights=local_effects, minlength=n_flat_bins
    )

    mean_effects_flat = np.zeros(n_flat_bins, dtype=float)
    non_zero = bin_counts_flat > 0
    mean_effects_flat[non_zero] = (
        bin_sums_flat[non_zero] / bin_counts_flat[non_zero]
    )

    bin_counts = bin_counts_flat.reshape(n_bins_i, n_bins_j)
    mean_effects = mean_effects_flat.reshape(n_bins_i, n_bins_j)

    # Double cumulative sum : uncentred 2D ALE at bin corners
    ale_inner = np.cumsum(np.cumsum(mean_effects, axis=0), axis=1)
    ale = np.zeros((n_bins_i + 1, n_bins_j + 1))
    ale[1:, 1:] = ale_inner

    # Remove main effects
    bin_counts_i = bin_counts.sum(axis=1)
    bin_counts_j = bin_counts.sum(axis=0)

    delta_i = ale[1:, 1:] - ale[:-1, 1:]
    delta_j = ale[1:, 1:] - ale[1:, :-1]

    main_effects_i = np.zeros(n_bins_i + 1)
    non_zero_i = bin_counts_i > 0
    main_effects_i_increments = np.zeros(n_bins_i)
    main_effects_i_increments[non_zero_i] = (
        np.sum(delta_i[non_zero_i, :] * bin_counts[non_zero_i, :], axis=1)
        / bin_counts_i[non_zero_i]
    )
    main_effects_i[1:] = np.cumsum(main_effects_i_increments)

    main_effects_j = np.zeros(n_bins_j + 1)
    non_zero_j = bin_counts_j > 0
    main_effects_j_increments = np.zeros(n_bins_j)
    main_effects_j_increments[non_zero_j] = (
        np.sum(delta_j[:, non_zero_j] * bin_counts[:, non_zero_j], axis=0)
        / bin_counts_j[non_zero_j]
    )
    main_effects_j[1:] = np.cumsum(main_effects_j_increments)

    ale -= main_effects_i[:, np.newaxis]
    ale -= main_effects_j[np.newaxis, :]

    # Centre: subtract the sample-weighted mean
    ale_centers = (
        (
            ale[:-1, :-1]  # ll
            + ale[1:, 1:]  # hh
            + ale[:-1, 1:]  # hl
            + ale[1:, :-1]  # lh
        )
        / 4.0
    )  # average of the four corner values

    # Centre: subtract the sample-weighted mean (only over occupied bins)
    ale -= np.sum(ale_centers * bin_counts) / bin_counts.sum()

    return {
        "ale": ale,
        "quantiles_i": quantiles_i,
        "quantiles_j": quantiles_j,
    }


class ALE:
    r"""
    Accumulated Local Effect (ALE) visualisation.
    :footcite:t:`apley2020accumulatedlocaleffects`

    ALE measures how the predictions of a model change on average when a
    feature varies locally within small intervals (bins). Unlike Partial
    Dependence Plots, ALE is not affected by feature correlations because it
    averages *local* differences rather than marginalising over the full
    feature distribution.

    Formally, for a single continuous feature :math:`x_j`, the 1D ALE corresponds to
    Equation (7) from :footcite:t:`apley2020accumulatedlocaleffects`:

    .. math::
        :label: eq_ale_1d_continuous

        \hat{f}_{j,\text{ALE}}(x) =
            \int_{z_{0,j}}^{x}
            \mathbb{E}\!\left[
                \frac{\partial f(X)}{\partial X_j}
                \;\middle|\; X_j = z
            \right] dz

    The centered 1D ALE curve :math:`\hat{f}_{j,\text{ALE}}(x_j)` is obtained by
    subtracting a constant :math:`c` so its weighted mean over the training
    distribution is zero: :math:`\hat{f}_{j,\text{ALE}}(x_j) = g_{j,\text{ALE}}(x_j) - c`.

    For a pair of continuous features :math:`(x_j, x_l)`, the uncentered 2D ALE interaction
    corresponds to Equation (11) from :footcite:t:`apley2020accumulatedlocaleffects`:

    .. math::
        :label: eq_ale_2d_continuous

        h_{\{j,l\},\text{ALE}}(x_j, x_l) =
            \int_{x_{\text{min},j}}^{x_j} \int_{x_{\text{min},l}}^{x_l}
            \mathbb{E}\!\left[
                \frac{\partial^2 f(X)}{\partial X_j \partial X_l}
                \;\middle|\; X_j = z_j, X_l = z_l
            \right] dz_j dz_l

    The final 2D ALE effect :math:`f_{\{j,l\},\text{ALE}}(x_j, x_l)` is then "doubly-centered"
    by subtracting the zero-order and first-order ALE main effects of :math:`X_j` and :math:`X_l`.

    Parameters
    ----------
    estimator : fitted sklearn-compatible estimator
        Must expose `predict`, `predict_proba`, or `decision_function`.
    feature_names : list of str, optional
        Names of the features. If None, X0, X1, ... will be used.

    Notes
    -----
    **Estimators and Discretization**

    In practice, the continuous derivatives and integrals are unknown. The package
    implements the local empirical estimators.

    For the 1D ALE estimator (approximating :eq:`eq_ale_1d_continuous` via Equation (9)
    of the reference paper):
    For each bin defined by the quantile grid of `X[:, feature_idx]`,
    the local effect is estimated as the average difference in model output
    when the feature moves from the lower to the upper bin edge across all
    samples that fall within that bin. The cumulative sum of these average
    effects gives the (uncentered) ALE curve, which is then centered so its
    weighted mean is zero.

    For the 2D ALE estimator (approximating :eq:`eq_ale_2d_continuous` via Equation (14)
    of the reference paper):
    The feature space of the pair is partitioned into a 2D grid of rectangular bins.
    The local interaction effect within a bin is estimated using second-order differences
    across the four corners of the bin for all instances falling into it. The uncentered
    interaction surface is computed by accumulating these local differences across both axes
    before performing the double-centering transformation.

    References
    ----------
    .. footbibliography::
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
        grid_resolution="auto",
        confidence_interval=False,
        confidence_level=0.95,
        percentiles=(0.05, 0.95),
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
        feature_type : string among "auto", "discrete", "continuous", or "categorical"
            Specify the type of values the feature has for 1D ALE. Set by default to auto and in this case :

            - non-numeric feature : categorical
            - numeric feature : discrete if the feature has less than 10 unique values or the number of unique values
              is less than 0.1% of the samples, and continuous otherwise
        grid_resolution : int or "auto", default="auto"
            Number of bins per feature axis. Set by default to "auto".

            - If "auto", the number of bins is determined automatically
              to minimize the histogram error per feature axis.
            - Note that the final number of bins in the quantile grids may be
              strictly less than `grid_resolution` (or the auto-calculated value)
              if the data contains many duplicate values or fewer unique points
              than requested.
        confidence_interval : bool, default=False
            Whether to compute and display confidence intervals around the 1D ALE curve.
        confidence_level : float, default=0.95
            The confidence level used to compute the confidence intervals (e.g., 0.95 for 95%).
        percentiles : tuple of float, default=(0.05, 0.95)
            The lower and upper percentile used to create the extreme values for the grid.
            Must be in [0, 1].
        cmap : str, default="viridis"
            Matplotlib colormap used for the 2D mesh plot.
        **kwargs
            Extra keyword arguments forwarded to:

            - `sns.lineplot` for 1D plots;
            - `ax.pcolormesh` for 2D plots.

        Returns
        -------
        axes : ndarray of Axes
            The matplotlib `Axes` objects that make up the figure.
        """
        X = np.asarray(X)
        mean_prediction = _predict_fn(self.estimator, X).mean()

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
                    confidence_interval=confidence_interval,
                    confidence_level=confidence_level,
                    percentiles=percentiles,
                )
            elif feature_type == "discrete":
                plotting_func = self._plot_1d_discrete
                result = compute_ale_1d_discrete(
                    self.estimator,
                    X,
                    feature_idx=features,
                    confidence_interval=confidence_interval,
                    confidence_level=confidence_level,
                    percentiles=percentiles,
                )
            else:
                raise ValueError(
                    "ALE (Accumulated Local Effects) is not supported for categorical features "
                    "because it requires a natural ordering to compute local differences. "
                    "Creating an artificial ordering would mislead the interpretation. "
                    "Please use alternative methods like M-plots or Partial Dependence Plots (PDP) instead."
                )
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
                percentiles=percentiles,
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
            mean_prediction=mean_prediction,
            cmap=cmap,
            **kwargs,
        )

    @staticmethod
    def _plot_1d_continuous(
        result,
        X,
        feature_ids,
        feature_names,
        mean_prediction,
        cmap=None,
        **kwargs,
    ):
        """Render a 1D continuous ALE plot with a marginal density strip."""
        del cmap  # only there for API compatibility

        feature_values = X[:, feature_ids[0]]
        low, high = result["quantiles"].min(), result["quantiles"].max()
        margin = (high - low) * 0.05

        _, axes = plt.subplots(2, 1, height_ratios=[0.2, 1], sharex=True)

        # Top strip: marginal distribution of the feature
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

        # Main panel: ALE curve
        ax_main = axes[1]
        sns.lineplot(
            x=result["quantiles"], y=result["ale"], ax=ax_main, **kwargs
        )
        if result["ale_err"] is not None:
            ax_main.fill_between(
                result["quantiles"],
                result["ale"] - result["ale_err"],
                result["ale"] + result["ale_err"],
                color="b",
                alpha=0.15,
                label="Confidence Interval",
            )
        ax_main.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax_main.set_xlim(low - margin, high + margin)
        ax_main.set_xlabel(feature_names[0])
        ax_main.set_ylabel("ALE (Centered)")

        ax_right = ax_main.twinx()
        ymin, ymax = ax_main.get_ylim()
        ax_right.set_ylim(ymin + mean_prediction, ymax + mean_prediction)
        ax_right.set_ylabel(
            "Mean modele prediction", rotation=270, labelpad=15
        )

        sns.despine(ax=ax_main, right=False)
        plt.tight_layout()
        return axes

    @staticmethod
    def _plot_1d_discrete(
        result,
        X,
        feature_ids,
        feature_names,
        mean_prediction,
        cmap=None,
        **kwargs,
    ):
        """Render a 1D discrete ALE plot with a marginal density histogram."""
        del cmap  # only there for API compatibility

        feature_values = X[:, feature_ids[0]]
        low, high = (
            result["unique_values"].min(),
            result["unique_values"].max(),
        )
        feature_values_filtered = feature_values[
            (feature_values >= low) & (feature_values <= high)
        ]

        _, axes = plt.subplots(
            2, 1, figsize=(8, 4), height_ratios=[0.2, 1], sharex=True
        )

        # Top strip: marginal distribution of the feature
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

        # Main panel: ALE curve
        ax_main = axes[1]
        sns.lineplot(
            x=result["unique_values"],
            y=result["ale"],
            ax=ax_main,
            marker="o",
            **kwargs,
        )
        if result["ale_err"] is not None:
            ax_main.fill_between(
                result["unique_values"],
                result["ale"] - result["ale_err"],
                result["ale"] + result["ale_err"],
                color="b",
                alpha=0.15,
            )
        ax_main.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax_main.xaxis.set_ticks(result["unique_values"])
        ax_main.set_xlim(low - 0.6, high + 0.6)
        ax_main.set_xlabel(feature_names[0])
        ax_main.set_ylabel("ALE (Centered)")

        ax_right = ax_main.twinx()
        ymin, ymax = ax_main.get_ylim()
        ax_right.set_ylim(ymin + mean_prediction, ymax + mean_prediction)
        ax_right.set_ylabel(
            "Mean modele prediction", rotation=270, labelpad=15
        )

        sns.despine(ax=ax_main, right=False)
        plt.tight_layout()
        return axes

    @staticmethod
    def _plot_2d(
        result,
        X,
        feature_ids,
        feature_names,
        mean_prediction,
        cmap="viridis",
        **kwargs,
    ):
        """Render a 2D ALE plot with marginal density strips on each axis."""
        x = X[:, feature_ids[0]]
        y = X[:, feature_ids[1]]

        quantiles_i = result["quantiles_i"]
        quantiles_j = result["quantiles_j"]
        ale = result["ale"]

        low_i, high_i = quantiles_i.min(), quantiles_i.max()
        low_j, high_j = quantiles_j.min(), quantiles_j.max()

        zz_cells = (
            ale[:-1, :-1] + ale[1:, 1:] + ale[:-1, 1:] + ale[1:, :-1]
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

        # Top strip: marginal of feature 0
        ax_top = axes[0, 0]
        sns.kdeplot(
            x, ax=ax_top, fill=True, color="black", alpha=0.25, legend=False
        )
        sns.despine(ax=ax_top, left=True)
        ax_top.xaxis.set_ticks([])
        ax_top.yaxis.set_visible(False)
        ax_top.set_xlim(low_i, high_i)

        # Right strip: marginal of feature 1
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

        # Color bar
        ax_cbar = axes[1, 4]
        cbar = fig.colorbar(mesh, cax=ax_cbar)
        cbar.set_label("ALE", rotation=90, labelpad=5)
        ax_cbar.yaxis.set_ticks_position("left")
        ax_cbar.yaxis.set_label_position("left")

        ax_cbar_right = ax_cbar.twinx()
        ymin, ymax = ax_cbar.get_ylim()
        ax_cbar_right.set_ylim(ymin + mean_prediction, ymax + mean_prediction)
        ax_cbar_right.set_ylabel(
            "Mean model prediction", rotation=270, labelpad=15
        )

        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.16, top=0.9)
        return axes
