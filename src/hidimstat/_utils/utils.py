import numbers
import numpy as np


def _check_vim_predict_method(method):
    """
    Validates that the method is a valid scikit-learn prediction method for variable importance measures.

    Parameters
    ----------
    method : str
        The scikit-learn prediction method to validate.

    Returns
    -------
    str
        The validated method if valid.

    Raises
    ------
    ValueError
        If the method is not one of the standard scikit-learn prediction methods:
        'predict', 'predict_proba' or 'decision_function'.
    """
    if method in ["predict", "predict_proba", "decision_function"]:
        return method
    else:
        raise ValueError(
            "The method {} is not a valid method "
            "for variable importance measure prediction".format(method)
        )


def check_random_state(seed):
    """
    Turn seed into a np.random.RandomState instance.
    This is based on the implementation of check_random_state of sciktilearn:
    https://github.com/scikit-learn/scikit-learn/blob/25dee604bae18205b01548348388baf7a1cdfe0e/sklearn/utils/validation.py#L1488

    Parameters
    ----------
    seed : None, int, tuple/list of 2 ints, or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is a tuple/list of 2 integers, creates a new seeded RandomState.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.

    Examples
    --------
    >>> from sklearn.utils.validation import check_random_state
    >>> check_random_state(42)
    RandomState(MT19937) at 0x...
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if (
        (isinstance(seed, tuple) or isinstance(seed, list))
        and len(seed) == 2
        and isinstance(seed[0], numbers.Integral)
        and isinstance(seed[1], numbers.Integral)
    ):
        return np.random.RandomState(np.random.default_rng(seed).bit_generator)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )


class SeedGenerator:
    """Generate seeds for parallel random number generation based on numpy guidelines.

    This class implements the recommended approach for parallel random number generation
    from numpy's documentation: https://numpy.org/doc/stable/reference/random/parallel.html

    Parameters
    ----------
    seed_root : int
        The root seed value to generate worker-specific seeds.
    """

    def __init__(self, seed_root):
        self.seed_root = seed_root

    def get_seed(self, worker_id):
        """Generate a seed pair for a specific worker.

        Parameters
        ----------
        worker_id : int
            Unique identifier for the worker/job.

        Returns
        -------
        list
            A list containing [worker_id, seed_root] for random number generation.
        """
        return [worker_id, self.seed_root]


class SequenceGenerator:
    """Generate seeds for parallel random number generation based on numpy guidelines.

    This class implements the recommended approach for parallel random number generation
    from numpy's documentation: https://numpy.org/doc/stable/reference/random/parallel.html

    Parameters
    ----------
    rgn : int or Generator
        Random number generator seed or instance to use as base generator.
    """

    def __init__(self, rgn):
        self.rng = np.random.default_rng(rgn.randint(np.iinfo(np.int32).max, size=1)[0])

    def get_seed(self, worker_id):
        """Generate a random state for a specific worker.

        Parameters
        ----------
        worker_id : int
            Unique identifier for the worker/job.

        Returns
        -------
        numpy.random.RandomState
            A RandomState instance seeded uniquely for this worker.
        """
        # Spawn a new independent generator and use its bits to seed a RandomState
        return np.random.RandomState(self.rng.spawn(1)[0].bit_generator)


class NoneGenerator:
    """Generator that always returns None as seed.

    This is used when no specific seed is required.
    """

    def get_seed(self, worker_id):
        """Return None regardless of worker_id.

        Parameters
        ----------
        worker_id : int
            Unique identifier for the worker/job (unused).

        Returns
        -------
        None
            Always returns None.
        """
        return None


def get_seed_generator(random_state):
    """
    Create appropriate seed generator based on input type.
    WARNING: this function should have the same branch that check_random_state

    Parameters
    ----------
    random_state : None, int, tuple/list of 2 ints, or numpy.random.RandomState
        The random state to use for seed generation.

    Returns
    -------
    SeedGenerator or NoneGenerator
        An appropriate generator for creating seeds.

    Raises
    ------
    ValueError
        If random_state is not of a supported type.
    """
    if random_state is None:
        return NoneGenerator()
    elif isinstance(random_state, numbers.Integral):
        return SeedGenerator(seed_root=random_state)
    elif (
        (isinstance(random_state, tuple) or isinstance(random_state, list))
        and len(random_state) == 2
        and isinstance(random_state[0], numbers.Integral)
        and isinstance(random_state[1], numbers.Integral)
    ):
        seed_root = check_random_state(random_state).randint(
            np.iinfo(np.int32).max, size=1
        )[0]
        return SeedGenerator(seed_root=seed_root)
    elif isinstance(random_state, np.random.RandomState):
        return SequenceGenerator(random_state)
    else:
        raise ValueError(
            "%r cannot be used to seed a numpy.random.RandomState instance"
            % random_state
        )
