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
        'predict', 'predict_proba', 'decision_function', or 'transform'.
    """
    if method in ["predict", "predict_proba", "decision_function", "transform"]:
        return method
    else:
        raise ValueError(
            "The method {} is not a valid method "
            "for variable importance measure prediction".format(method)
        )


def check_random_state(seed):
    """
    Modified version of sklearn's check_random_state using np.random.Generator.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None or seed is np.random:
        return np.random.default_rng()
    if isinstance(seed, numbers.Integral):
        return np.random.default_rng(seed)
    raise ValueError(
        "%r cannot be used to seed a numpy.random.Generator instance" % seed
    )
