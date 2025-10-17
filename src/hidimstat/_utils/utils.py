import numbers
from functools import partial

import numpy as np
from numpy.random import RandomState
from scipy.stats import ttest_1samp, wilcoxon


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


def get_fitted_attributes(cls):
    """
    Get all attributes from a class that end with a single underscore
    and doesn't start with one underscore.

    Parameters
    ----------
    cls : class
        The class to inspect for attributes.

    Returns
    -------
    list
        A list of attribute names that end with a single underscore but not double underscore.
    """
    # Get all attributes and methods of the class
    all_attributes = dir(cls)

    # Filter out attributes that start with an underscore
    filtered_attributes = [attr for attr in all_attributes if not attr.startswith("_")]

    # Filter out attributes that do not end with a single underscore
    result = [
        attr
        for attr in filtered_attributes
        if attr.endswith("_") and not attr.endswith("__")
    ]

    return result


def check_random_state(seed):
    """
    Modified version of sklearn's check_random_state using np.random.Generator.
    This is based on the implementation of check_random_state of sciktilearn:
    https://github.com/scikit-learn/scikit-learn/blob/25dee604bae18205b01548348388baf7a1cdfe0e/sklearn/utils/validation.py#L1488

    Parameters
    ----------
    seed : None, int or Generator
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a Generator instance, return it.
        If seed is a RandomState instance, raise an exception.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.Generator`
        The random state object based on `seed` parameter.

    Examples
    --------
    >>> from sklearn.utils.validation import check_random_state
    >>> check_random_state(42)
    BitGenerator (PCG64) at 0x...
    """
    if seed is None or seed is np.random:
        return np.random.default_rng()
    if isinstance(seed, numbers.Integral):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, RandomState):
        raise ValueError(
            "numpy.random.RandomState is deprecated. Please use numpy.random.Generator "
            "instead by calling numpy.random.default_rng(seed). See "
            "https://numpy.org/doc/stable/reference/random/generator.html for details."
        )
    raise ValueError(
        "%r cannot be used to seed a numpy.random.Generator instance" % seed
    )


def seed_estimator(estimator, random_state=None):
    """
    Sets the random_state for a scikit-learn estimator and its components.

    This function inspects the estimator and any of its attributes, setting their
    `random_state` attribute to the provided value.

    Parameters
    ----------
    estimator : sklearn estimator
        The scikit-learn estimator to seed.
    random_state : int, Generator, or None, default=None
        The random state to set. If None, the estimator's random_state is not changed.

    Returns
    -------
    The seeded estimator object.
    """
    rng = check_random_state(random_state)
    # Set the random_state of the main estimator
    if hasattr(estimator, "random_state"):
        estimator.set_params(random_state=RandomState(rng.bit_generator))

    if hasattr(estimator, "__dict__"):
        for _, value in estimator.__dict__.items():
            if hasattr(value, "random_state"):
                setattr(value, "random_state", RandomState(rng.bit_generator))

    return estimator


def check_statistical_test(statistical_test):
    """
    Validates and returns a test statistic function.

    Parameters
    ----------
    test : str or callable
        If str, must be either 'ttest' or 'wilcoxon'.
        If callable, must be a function that can be used as a test statistic.

    Returns
    -------
    callable
        A function that can be used as a test statistic.
        For string inputs, returns a partial function of either ttest_1samp or wilcoxon.
        For callable inputs, returns the input function.

    Raises
    ------
    ValueError
        If test is a string but not one of the supported test names ('ttest' or 'wilcoxon').
    ValueError
        If test is neither a string nor a callable.
    """
    if isinstance(statistical_test, str):
        if statistical_test == "ttest":
            return partial(ttest_1samp, popmean=0, alternative="greater", axis=1)
        elif statistical_test == "wilcoxon":
            return partial(wilcoxon, alternative="greater", axis=1)
        else:
            raise ValueError(f"the test '{statistical_test}' is not supported")
    elif callable(statistical_test):
        return statistical_test
    else:
        raise ValueError(
            f"Unsupported value for 'statistical_test'."
            f"The provided argument was '{statistical_test}'. "
            f"Please choose from the following valid options: "
            f"string values ('ttest', 'wilcoxon', 'NB-ttest') "
            f"or a custom callable function with a `scipy.stats` API-compatible signature."
        )
