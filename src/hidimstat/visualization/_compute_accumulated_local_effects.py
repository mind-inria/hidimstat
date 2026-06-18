import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def _check_estimator_predict(estimator):
    """Return a scalar-output prediction callable for *estimator*.

    For classifiers we use ``predict_proba`` (returns the probability of the
    positive class for binary problems) when available, otherwise
    ``decision_function``.  For regressors we use ``predict``.

    Parameters
    ----------
    estimator : fitted sklearn-compatible estimator

    Returns
    -------
    predict_fn : callable
        A function that accepts an array of shape ``(n_samples, n_features)``
        and returns a 1D array of shape ``(n_samples,)``.
    """
    feature_names = getattr(estimator, "feature_names_in_", None)

    if hasattr(estimator, "predict_proba"):

        def predict_fn(X):
            if feature_names is not None:
                X = pd.DataFrame(X, columns=feature_names)
            proba = estimator.predict_proba(X)
            # Binary: keep only the positive-class column
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba

        return predict_fn

    if hasattr(estimator, "decision_function"):

        def predict_fn(X):
            if feature_names is not None:
                X = pd.DataFrame(X, columns=feature_names)
            return estimator.decision_function(X).ravel()

        return predict_fn

    if hasattr(estimator, "predict"):

        def predict_fn(X):
            if feature_names is not None:
                X = pd.DataFrame(X, columns=feature_names)
            return estimator.predict(X).ravel()

        return predict_fn

    raise ValueError(
        "'estimator' must expose at least one of predict_proba, "
        "decision_function, or predict."
    )


def _build_quantile_grid(x, grid_resolution):
    """Build a 1D quantile grid for a single continuous feature column.

    Parameters
    ----------
    x : ndarray of shape (n_samples,)
        Values for one feature.
    grid_resolution : int
        Number of bin in the grid.

    Returns
    -------
    quantiles : ndarray of shape (n_quantiles,)
        Unique, sorted quantile bins edges.
    """
    if grid_resolution <= 0:
        raise ValueError("'grid_resolution' must be strictly greater than 0.")

    # Use unique values when there are fewer than grid_resolution unique points
    uniques = np.unique(x[~np.isnan(x)])
    if uniques.shape[0] <= grid_resolution:
        return uniques

    probs = np.linspace(0.0, 1.0, grid_resolution + 1)
    return np.unique(np.percentile(x, probs * 100, method="inverted_cdf"))


def _bin_indices(x, quantiles):
    """Assign each sample to a bin defined by *quantiles*.

    Samples are placed in bin ``k`` when ``quantiles[k] <= x < quantiles[k+1]``.
    The last bin is closed on the right.  Indices are clipped so that every
    sample falls within ``[0, len(quantiles) - 2]``.

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


def compute_ale_1d_continuous(estimator, X, feature_idx, grid_resolution=50):
    """Compute the 1D Accumulated Local Effect for a single continuous feature.

    For each bin defined by the quantile grid of ``X[:, feature_idx]``,
    the local effect is estimated as the average difference in model output
    when the feature moves from the lower to the upper bin edge across all
    samples that fall within that bin. The cumulative sum of these average
    effects gives the (uncentered) ALE curve, which is then centered so its
    weighted mean is zero.

    Parameters
    ----------
    estimator : fitted sklearn-compatible estimator
        Must expose ``predict``, ``predict_proba``, or ``decision_function``.
    X : array-like of shape (n_samples, n_features)
        Training (or evaluation) dataset used to define the quantile grid and
        to gather local samples in each bin.
    feature_idx : int
        Column index of the feature of interest.
    grid_resolution : int, default=50
        Number of bins used to build the quantile grid.

    Returns
    -------
    result : dict with the following keys:

        ``"ale"`` : ndarray of shape (n_quantiles,)
            ALE values evaluated at each bin edge.
        ``"quantiles"`` : ndarray of shape (n_quantiles,)
            Bin edges (quantiles of the feature distribution).
    """
    X = np.asarray(X)
    predict_fn = _check_estimator_predict(estimator)

    x = X[:, feature_idx]
    quantiles = _build_quantile_grid(x, grid_resolution)
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

    local_effects = predict_fn(X_high) - predict_fn(
        X_low
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

    return {
        "ale": ale,
        "quantiles": quantiles,
    }


def compute_ale_1d_discrete(estimator, X, feature_idx):
    """Compute the 1D Accumulated Local Effect for a single discrete feature.

    For each value of the feature, the local effect is estimated as the average
    difference in model output when the feature moves from the lower to the upper
    neighbour values across all samples that have that value. The cumulative sum
    of these average effects gives the (uncentered) ALE curve, which is then
    centered so its weighted mean is zero.

    Parameters
    ----------
    estimator : fitted sklearn-compatible estimator
        Must expose ``predict``, ``predict_proba``, or ``decision_function``.
    X : array-like of shape (n_samples, n_features)
        Training (or evaluation) dataset used to define the quantile grid and
        to gather local samples in each bin.
    feature_idx : int
        Column index of the feature of interest.

    Returns
    -------
    result : dict with the following keys:

        ``"ale"`` : ndarray of shape (n_quantiles,)
            ALE values evaluated at each value of the feature.
        ``"unique_values"`` : ndarray of shape (n_values,)
            Unique values of the feature of interest
    """
    X = np.asarray(X)
    predict_fn = _check_estimator_predict(estimator)

    x = X[:, feature_idx]
    unique_values = np.unique(x)
    n_values = len(unique_values)

    if n_values < 1:
        raise ValueError(
            f"Feature {feature_idx} has fewer than 2 unique values. Check your data."
        )

    value_idx = np.digitize(x, unique_values) - 1

    # For each sample, evaluate the model at the lower and upper edge of its bin
    X_low = X.copy()
    X_high = X.copy()
    mask_low = x != unique_values[0]
    mask_high = x != unique_values[-1]
    X_low[mask_low, feature_idx] = unique_values[value_idx[mask_low] - 1]
    X_high[mask_high, feature_idx] = unique_values[value_idx[mask_high] + 1]

    local_effects_low = predict_fn(X[mask_low]) - predict_fn(X_low[mask_low])
    local_effects_high = predict_fn(X_high[mask_high]) - predict_fn(
        X[mask_high]
    )

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

    return {"ale": ale, "unique_values": unique_values}


def compute_ale_2d(
    estimator,
    X,
    feature_indices,
    grid_resolution=20,
    is_categorical=(False, False),
    impute_empty_bins=False,
):
    """Compute the 2D Accumulated Local Effect for a pair of continuous features.

    The 2D ALE isolates the *interaction* effect between two features by
    accumulating second-order local differences across a 2D grid of quantile
    bins.  The resulting surface captures the joint effect that cannot be
    explained by either 1D ALE alone.

    Parameters
    ----------
    estimator : fitted sklearn-compatible estimator
    X : array-like of shape (n_samples, n_features)
    feature_indices : tuple or list of two ints
        Column indices ``[i, j]`` of the two features of interest.
    grid_resolution : int, default=20
        Number of bins per feature axis.
    is_categorical : tuple or list of two bools, default=(False, False)
        Reserved for future categorical support.  Both values must be ``False``.
    impute_empty_bins : bool, default=True
        If ``True``, empty bins (no training samples) are filled with the
        mean effect of their nearest non-empty neighbour in normalised feature
        space before the cumulative sum is computed.  This avoids the zero-
        padding bias that would otherwise propagate through the double cumsum
        in sparse regions of the feature space.


    Returns
    -------
    result : dict with the following keys:

        ``"ale"`` : ndarray of shape (n_quantiles_i, n_quantiles_j)
            2D ALE values evaluated at each 2D bin corner.
        ``"quantiles_i"`` : ndarray of shape (n_quantiles_i,)
            Bin edges for the first feature.
        ``"quantiles_j"`` : ndarray of shape (n_quantiles_j,)
            Bin edges for the second feature.
    """
    if any(is_categorical):
        raise NotImplementedError(
            "Categorical features are not yet supported. Set is_categorical=(False, False)."
        )

    feature_indices = list(feature_indices)
    if len(feature_indices) != 2:
        raise ValueError(
            "feature_indices must contain exactly two feature indices."
        )

    X = np.asarray(X)
    predict_fn = _check_estimator_predict(estimator)

    idx_i, idx_j = feature_indices
    x_i, x_j = X[:, idx_i], X[:, idx_j]

    quantiles_i = _build_quantile_grid(x_i, grid_resolution)
    quantiles_j = _build_quantile_grid(x_j, grid_resolution)

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

    # Second-order finite differences: evaluate at the four corners of each 2D bin
    X_ll = X.copy()
    X_ll[:, idx_i] = quantiles_i[bin_idx_i]
    X_ll[:, idx_j] = quantiles_j[bin_idx_j]

    X_lh = X.copy()
    X_lh[:, idx_i] = quantiles_i[bin_idx_i]
    X_lh[:, idx_j] = quantiles_j[bin_idx_j + 1]

    X_hl = X.copy()
    X_hl[:, idx_i] = quantiles_i[bin_idx_i + 1]
    X_hl[:, idx_j] = quantiles_j[bin_idx_j]

    X_hh = X.copy()
    X_hh[:, idx_i] = quantiles_i[bin_idx_i + 1]
    X_hh[:, idx_j] = quantiles_j[bin_idx_j + 1]

    # Second-order local effects
    local_effects = (predict_fn(X_hh) - predict_fn(X_hl)) - (
        predict_fn(X_lh) - predict_fn(X_ll)
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

    if impute_empty_bins:
        raise ValueError("Not yet implemented.")

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
