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


def get_generated_attributes(cls):
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
