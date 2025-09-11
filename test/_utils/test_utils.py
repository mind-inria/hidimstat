import numbers
import numpy as np
import pytest
from hidimstat._utils.utils import (
    get_seed_generator,
    check_random_state,
    NoneGenerator,
    SeedGenerator,
    SequenceGenerator,
)


def test_none():
    "test random state is None"
    random_state = None
    check_random_state(random_state)
    generator = get_seed_generator(random_state)

    assert isinstance(generator, NoneGenerator)

    seed = generator.get_seed(52)
    assert seed is None


def test_integer():
    "test random state is integer"
    random_state = 10
    check_random_state(random_state)
    generator = get_seed_generator(random_state)

    assert isinstance(generator, SeedGenerator)

    seed = generator.get_seed(52)
    assert isinstance(seed, list)
    assert len(seed) == 2
    assert seed[0] == 52
    assert isinstance(seed[1], numbers.Integral)


def test_tuple_2_integer():
    "test random state is a tuple of 2 elements"
    random_state = (0, 1)
    check_random_state(random_state)
    generator = get_seed_generator(random_state)

    assert isinstance(generator, SeedGenerator)

    seed = generator.get_seed(52)
    assert isinstance(seed, list)
    assert len(seed) == 2
    assert seed[0] == 52
    assert isinstance(seed[1], numbers.Integral)


def test_array_2_integer():
    "test random state is an array of 2 elements"
    random_state = [0, 1]
    check_random_state(random_state)
    generator = get_seed_generator(random_state)

    assert isinstance(generator, SeedGenerator)

    seed = generator.get_seed(52)
    assert isinstance(seed, list)
    assert len(seed) == 2
    assert seed[0] == 52
    assert isinstance(seed[1], numbers.Integral)


def test_rng():
    "test random state is rng"
    random_state = np.random.RandomState(0)
    check_random_state(random_state)
    generator = get_seed_generator(random_state)

    assert isinstance(generator, SequenceGenerator)

    seed = generator.get_seed(52)
    assert isinstance(seed, np.random.RandomState)


def test_error():
    "test random state is rng"
    random_state = [1, 2, 3]
    with pytest.raises(
        ValueError, match="cannot be used to seed a numpy.random.RandomState instance"
    ):
        check_random_state(random_state)
    with pytest.raises(
        ValueError, match="cannot be used to seed a numpy.random.RandomState instance"
    ):
        generator = get_seed_generator(random_state)
