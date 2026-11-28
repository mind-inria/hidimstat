import numpy as np


def _build_quantile_grid_1d(x, grid_resolution, percentiles=(5, 95)):
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

    percentiles : tuple of float, default=(5, 95)
        The lower and upper percentile used to create the extreme values for the grid.
        Must be in [0, 100].


    Returns
    -------
    quantiles : ndarray of shape (n_quantiles,)
        Unique, sorted quantile bins edges.
    """
    if (
        not isinstance(percentiles, tuple)
        or len(percentiles) != 2
        or not (0 <= percentiles[0] <= percentiles[1] <= 100)
    ):
        raise ValueError(
            "'percentiles' must be a tuple of 2 floats "
            "in [0, 100] in increasing order"
        )

    low_bnd = np.percentile(x, percentiles[0])
    high_bnd = np.percentile(x, percentiles[1])

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


def _build_quantile_grid_2d(x, y, grid_resolution, percentiles=(5, 95)):
    """Build a 2D quantile grid for a pair of continuous features.

    Parameters
    ----------
    x : ndarray of shape (n_samples,)
        Values for the first feature.
    y : ndarray of shape (n_samples,)
        Values for the second feature.
    grid_resolution : int, tuple of int, or "auto", default="auto"
        Number of bins in the grid. Set by default to "auto".

        - If "auto", the number of bins for each feature is determined
          automatically using Scott's rule for a bivariate normal distribution,
          accounting for sample size, standard deviations, and correlation.
        - If an int, the same number of bins is applied to both features.
        - If a tuple of 2 ints, specifies (grid_resolution_x, grid_resolution_y).
        - Note that the final number of bins in the returned grids may be
          strictly less than the requested or auto-calculated value if the data
          contains many duplicate values or fewer unique points than requested.

    percentiles : tuple of float, default=(5, 95)
        The lower and upper percentile used to create the extreme values for the grid.
        Must be in [0, 100].

    Returns
    -------
    quantiles_x : ndarray of shape (n_quantiles_x,)
        Unique, sorted quantile bin edges for the first feature.
    quantiles_y : ndarray of shape (n_quantiles_y,)
        Unique, sorted quantile bin edges for the second feature.
    """
    if (
        not isinstance(percentiles, tuple)
        or len(percentiles) != 2
        or not (0 <= percentiles[0] <= percentiles[1] <= 100)
    ):
        raise ValueError(
            "'percentiles' must be a tuple of 2 floats "
            "in [0, 100] in increasing order"
        )

    if grid_resolution == "auto":
        low_x, high_x = (
            np.percentile(x, percentiles[0]),
            np.percentile(x, percentiles[1]),
        )
        low_y, high_y = (
            np.percentile(y, percentiles[0]),
            np.percentile(y, percentiles[1]),
        )

        joint_mask = (
            (x >= low_x) & (x <= high_x) & (y >= low_y) & (y <= high_y)
        )
        x_filtered = x[joint_mask]
        y_filtered = y[joint_mask]
        n = len(x_filtered)

        if n > 1:
            sigma_x = np.std(x_filtered)
            sigma_y = np.std(y_filtered)

            if sigma_x > 0 and sigma_y > 0:
                rho = np.corrcoef(x_filtered, y_filtered)[0, 1]
                rho = np.clip(rho, -1, 1)
            else:
                rho = 0

            factor = 3.504 * ((1.0 - rho**2) ** (3 / 8)) * (n ** (-1 / 4))

            h_x = factor * sigma_x if sigma_x > 0 else 1
            h_y = factor * sigma_y if sigma_y > 0 else 1

            range_x = high_x - low_x
            range_y = high_y - low_y

            res_x = (
                max(1, int(np.ceil(range_x / h_x)))
                if range_x > 0 and h_x > 0
                else 1
            )
            res_y = (
                max(1, int(np.ceil(range_y / h_y)))
                if range_y > 0 and h_y > 0
                else 1
            )
        else:
            res_x = res_y = 1
    elif isinstance(grid_resolution, tuple):
        if len(grid_resolution) != 2 or not all(
            isinstance(r, (int, np.integer)) and r > 0 for r in grid_resolution
        ):
            raise ValueError(
                "If 'grid_resolution' is a tuple, it must contain exactly 2 strictly positive integers."
            )
        res_x, res_y = grid_resolution
    elif (
        isinstance(grid_resolution, (int, np.integer)) and grid_resolution > 0
    ):
        res_x = res_y = grid_resolution
    else:
        raise ValueError(
            "'grid_resolution' must be 'auto', an int strictly greater than 0, or a tuple of 2 int strictly greater than 0."
        )

    quantiles_x = _build_quantile_grid_1d(
        x, grid_resolution=res_x, percentiles=percentiles
    )
    quantiles_y = _build_quantile_grid_1d(
        y, grid_resolution=res_y, percentiles=percentiles
    )

    return quantiles_x, quantiles_y


def _bin_indices(x, quantiles):
    """Assign each sample to a bin defined by *quantiles*.

    Samples are placed in bin `k` when `quantiles[k] <= x < quantiles[k+1]`.
    The last bin is closed on the right. Indices are clipped so that every
    sample falls within `[0, len(quantiles) - 2]`.

    Parameters
    ----------
    x : ndarray of shape (n_samples,)
        The 1D array containing the features values of the samples to bin.
    quantiles : ndarray of shape (n_quantiles,)
        The 1D array of ordered bin edges.

    Returns
    -------
    indices : ndarray of shape (n_samples,), dtype int
    """
    # digitize returns 0 for x < quantiles[0] and len(quantiles) for x > quantiles[-1]
    idx = np.digitize(x, quantiles) - 1
    # samples equal to the last edge fall in the last bin
    return np.clip(idx, 0, len(quantiles) - 2)
