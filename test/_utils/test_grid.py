import numpy as np
import pytest

from hidimstat._utils.grid import (
    _bin_indices,
    _build_quantile_grid_1d,
    _build_quantile_grid_2d,
)


def test_build_quantile_grid_1d():
    """Test 1D quantile grid creation."""
    x = np.arange(100)

    grid_auto = _build_quantile_grid_1d(x, "auto")
    assert len(grid_auto) > 1

    with pytest.raises(
        ValueError, match="'percentiles' must be a tuple of 2 floats"
    ):
        _build_quantile_grid_1d(x, 0, percentiles=[5, 95])
    with pytest.raises(
        ValueError, match="'percentiles' must be a tuple of 2 floats"
    ):
        _build_quantile_grid_1d(x, 0, percentiles=(5, 25, 50, 75, 95))
    with pytest.raises(
        ValueError, match="'percentiles' must be a tuple of 2 floats"
    ):
        _build_quantile_grid_1d(x, 0, percentiles=(-100, 50))
    with pytest.raises(
        ValueError, match="'percentiles' must be a tuple of 2 floats"
    ):
        _build_quantile_grid_1d(x, 0, percentiles=(50, -100))

    with pytest.raises(
        ValueError,
        match="'grid_resolution' must be an int strictly greater than 0",
    ):
        _build_quantile_grid_1d(x, "invalid_resolution")
    with pytest.raises(
        ValueError,
        match="'grid_resolution' must be an int strictly greater than 0",
    ):
        _build_quantile_grid_1d(x, 0)
    with pytest.raises(
        ValueError,
        match="'grid_resolution' must be an int strictly greater than 0",
    ):
        _build_quantile_grid_1d(x, -5)

    x_few = np.array([1, 1, 2, 2, 3])
    grid_few = _build_quantile_grid_1d(
        x_few, grid_resolution=10, percentiles=(0, 100)
    )
    np.testing.assert_array_equal(grid_few, [1, 2, 3])


def test_build_quantile_grid_2d():
    """Test 2D quantile grid creation."""
    x = np.arange(100)
    y = np.arange(100) * 2

    grid_auto_x, grid_auto_y = _build_quantile_grid_2d(x, y, "auto")
    assert len(grid_auto_x) > 1
    assert len(grid_auto_y) > 1

    grid_int_x, grid_int_y = _build_quantile_grid_2d(x, y, 10)
    assert len(grid_int_x) > 1
    assert len(grid_int_y) > 1

    grid_tuple_x, grid_tuple_y = _build_quantile_grid_2d(x, y, (5, 8))
    assert len(grid_tuple_x) > 1
    assert len(grid_tuple_y) > 1

    # Percentiles errors
    with pytest.raises(
        ValueError, match="'percentiles' must be a tuple of 2 floats"
    ):
        _build_quantile_grid_2d(x, y, "auto", percentiles=[5, 95])
    with pytest.raises(
        ValueError, match="'percentiles' must be a tuple of 2 floats"
    ):
        _build_quantile_grid_2d(x, y, "auto", percentiles=(5, 25, 50, 75, 95))
    with pytest.raises(
        ValueError, match="'percentiles' must be a tuple of 2 floats"
    ):
        _build_quantile_grid_2d(x, y, "auto", percentiles=(-100, 50))
    with pytest.raises(
        ValueError, match="'percentiles' must be a tuple of 2 floats"
    ):
        _build_quantile_grid_2d(x, y, "auto", percentiles=(50, -100))

    # Errors when grid_resolution is an integer/"auto"
    with pytest.raises(
        ValueError, match="'grid_resolution' must be 'auto', an int"
    ):
        _build_quantile_grid_2d(x, y, "invalid_grid_resolution")
    with pytest.raises(
        ValueError, match="'grid_resolution' must be 'auto', an int"
    ):
        _build_quantile_grid_2d(x, y, 0)
    with pytest.raises(
        ValueError, match="'grid_resolution' must be 'auto', an int"
    ):
        _build_quantile_grid_2d(x, y, -5)

    # Errors when grid_resolution is a tuple
    with pytest.raises(
        ValueError, match="must contain exactly 2 strictly positive integers"
    ):
        _build_quantile_grid_2d(x, y, (5, "invalid_grid_resolution"))
    with pytest.raises(
        ValueError, match="must contain exactly 2 strictly positive integers"
    ):
        _build_quantile_grid_2d(x, y, (5,))
    with pytest.raises(
        ValueError, match="must contain exactly 2 strictly positive integers"
    ):
        _build_quantile_grid_2d(x, y, (5, 10, 15))
    with pytest.raises(
        ValueError, match="must contain exactly 2 strictly positive integers"
    ):
        _build_quantile_grid_2d(x, y, (5, -2))
    with pytest.raises(
        ValueError, match="must contain exactly 2 strictly positive integers"
    ):
        _build_quantile_grid_2d(x, y, (0, 5))

    # Limit n <= 1
    x_small = np.array([1])
    y_small = np.array([1])
    grid_small_x, grid_small_y = _build_quantile_grid_2d(
        x_small, y_small, "auto"
    )
    np.testing.assert_array_equal(grid_small_x, [1])
    np.testing.assert_array_equal(grid_small_y, [1])


def test_bin_indices():
    """Verify that the bin assignment correctly handles extreme values ​​via clip."""
    quantiles = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])
    indices = _bin_indices(x, quantiles)
    np.testing.assert_array_equal(indices, [0, 0, 1, 2, 2])
