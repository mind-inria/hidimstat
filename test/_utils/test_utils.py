import numpy as np
import pytest
from hidimstat._utils.utils import check_random_state


def test_none():
    "test random state is None"
    random_state = None
    rng = check_random_state(random_state)
    assert isinstance(rng, np.random.Generator)


def test_integer():
    "test random state is integer"
    random_state = 10
    rng = check_random_state(random_state)
    assert isinstance(rng, np.random.Generator)


def test_rng():
    "test random state is rng"
    random_state = np.random.RandomState(0)
    with pytest.raises(
        ValueError,
        match="numpy.random.RandomState is deprecated. Please use numpy.random.Generator",
    ):
        _ = check_random_state(random_state)


def test_error():
    "test random state is rng"
    random_state = [1, 2, 3]
    with pytest.raises(
        ValueError, match="cannot be used to seed a numpy.random.Generator instance"
    ):
        check_random_state(random_state)
