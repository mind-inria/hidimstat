from collections import namedtuple
from keyword import iskeyword as _iskeyword
import sys as _sys

import numpy as np
import scipy.special as special
from scipy.stats._stats_py import _var


class _SimpleStudentT:
    """
    Based on https://github.com/scipy/scipy/blob/e407bc4d6ee71ec1a23fce9d95b58e28451e4d94/scipy/stats/_stats_py.py#L10886
    """

    # A very simple, array-API compatible t distribution for use in
    # hypothesis tests. May be replaced by new infrastructure t
    # distribution in due time.
    def __init__(self, df):
        self.df = df

    def cdf(self, t):
        return special.stdtr(self.df, t)

    def sf(self, t):
        return special.stdtr(self.df, -t)


## definition of Confidence interval
## Based on https://github.com/scipy/scipy/blob/e407bc4d6ee71ec1a23fce9d95b58e28451e4d94/scipy/stats/_common.py#L4
ConfidenceInterval = namedtuple("ConfidenceInterval", ["low", "high"])
ConfidenceInterval.__doc__ = "Class for confidence intervals."


def _t_confidence_interval(df, t, confidence_level, alternative, dtype=None, xp=None):
    """
    Based on https://github.com/scipy/scipy/blob/e407bc4d6ee71ec1a23fce9d95b58e28451e4d94/scipy/stats/_stats_py.py#L6268
    """
    # Input validation on `alternative` is already done
    # We just need IV on confidence_level
    dtype = t.dtype if dtype is None else dtype

    if confidence_level < 0 or confidence_level > 1:
        message = "`confidence_level` must be a number between 0 and 1."
        raise ValueError(message)

    confidence_level = np.asarray(confidence_level, dtype=dtype)
    inf = np.asarray(np.inf, dtype=dtype)

    if alternative < 0:  # 'less'
        p = confidence_level
        low, high = np.broadcast_arrays(-inf, special.stdtrit(df, p))
    elif alternative > 0:  # 'greater'
        p = 1 - confidence_level
        low, high = np.broadcast_arrays(special.stdtrit(df, p), inf)
    elif alternative == 0:  # 'two-sided'
        tail_probability = (1 - confidence_level) / 2
        p = np.stack((tail_probability, 1 - tail_probability))
        # axis of p must be the zeroth and orthogonal to all the rest
        p = np.reshape(p, tuple([2] + [1] * np.asarray(df).ndim))
        ci = special.stdtrit(df, p)
        low, high = ci[0, ...], ci[1, ...]
    else:  # alternative is NaN when input is empty (see _axis_nan_policy)
        nan = np.asarray(np.nan)
        p, nans = np.broadcast_arrays(t, nan)
        low, high = nans, nans

    low = np.asarray(low, dtype=dtype)
    low = low[()] if low.ndim == 0 else low
    high = np.asarray(high, dtype=dtype)
    high = high[()] if high.ndim == 0 else high
    return low, high


def _validate_names(typename, field_names, extra_field_names):
    """
    Based on https://github.com/scipy/scipy/blob/e407bc4d6ee71ec1a23fce9d95b58e28451e4d94/scipy/_lib/_bunch.py#L5

    Ensure that all the given names are valid Python identifiers that
    do not start with '_'.  Also check that there are no duplicates
    among field_names + extra_field_names.
    """
    for name in [typename] + field_names + extra_field_names:
        if not isinstance(name, str):
            raise TypeError("typename and all field names must be strings")
        if not name.isidentifier():
            raise ValueError(
                "typename and all field names must be valid " f"identifiers: {name!r}"
            )
        if _iskeyword(name):
            raise ValueError(
                "typename and all field names cannot be a " f"keyword: {name!r}"
            )

    seen = set()
    for name in field_names + extra_field_names:
        if name.startswith("_"):
            raise ValueError(
                "Field names cannot start with an underscore: " f"{name!r}"
            )
        if name in seen:
            raise ValueError(f"Duplicate field name: {name!r}")
        seen.add(name)


# Note: This code is adapted from CPython:Lib/collections/__init__.py
def _make_tuple_bunch(typename, field_names, extra_field_names=None, module=None):
    """
    Create a namedtuple-like class with additional attributes.

    Based on https://github.com/scipy/scipy/blob/e407bc4d6ee71ec1a23fce9d95b58e28451e4d94/scipy/_lib/_bunch.py#L32

    This function creates a subclass of tuple that acts like a namedtuple
    and that has additional attributes.

    The additional attributes are listed in `extra_field_names`.  The
    values assigned to these attributes are not part of the tuple.

    The reason this function exists is to allow functions in SciPy
    that currently return a tuple or a namedtuple to returned objects
    that have additional attributes, while maintaining backwards
    compatibility.

    This should only be used to enhance *existing* functions in SciPy.
    New functions are free to create objects as return values without
    having to maintain backwards compatibility with an old tuple or
    namedtuple return value.

    Parameters
    ----------
    typename : str
        The name of the type.
    field_names : list of str
        List of names of the values to be stored in the tuple. These names
        will also be attributes of instances, so the values in the tuple
        can be accessed by indexing or as attributes.  At least one name
        is required.  See the Notes for additional restrictions.
    extra_field_names : list of str, optional
        List of names of values that will be stored as attributes of the
        object.  See the notes for additional restrictions.

    Returns
    -------
    cls : type
        The new class.

    Notes
    -----
    There are restrictions on the names that may be used in `field_names`
    and `extra_field_names`:

    * The names must be unique--no duplicates allowed.
    * The names must be valid Python identifiers, and must not begin with
      an underscore.
    * The names must not be Python keywords (e.g. 'def', 'and', etc., are
      not allowed).
    """
    if len(field_names) == 0:
        raise ValueError("field_names must contain at least one name")

    if extra_field_names is None:
        extra_field_names = []
    _validate_names(typename, field_names, extra_field_names)

    typename = _sys.intern(str(typename))
    field_names = tuple(map(_sys.intern, field_names))
    extra_field_names = tuple(map(_sys.intern, extra_field_names))

    all_names = field_names + extra_field_names
    arg_list = ", ".join(field_names)
    full_list = ", ".join(all_names)
    repr_fmt = "".join(
        ("(", ", ".join(f"{name}=%({name})r" for name in all_names), ")")
    )
    tuple_new = tuple.__new__
    _dict, _tuple, _zip = dict, tuple, zip

    # Create all the named tuple methods to be added to the class namespace

    s = f"""\
def __new__(_cls, {arg_list}, **extra_fields):
    return _tuple_new(_cls, ({arg_list},))

def __init__(self, {arg_list}, **extra_fields):
    for key in self._extra_fields:
        if key not in extra_fields:
            raise TypeError("missing keyword argument '%s'" % (key,))
    for key, val in extra_fields.items():
        if key not in self._extra_fields:
            raise TypeError("unexpected keyword argument '%s'" % (key,))
        self.__dict__[key] = val

def __setattr__(self, key, val):
    if key in {repr(field_names)}:
        raise AttributeError("can't set attribute %r of class %r"
                             % (key, self.__class__.__name__))
    else:
        self.__dict__[key] = val
"""
    del arg_list
    namespace = {
        "_tuple_new": tuple_new,
        "__builtins__": dict(TypeError=TypeError, AttributeError=AttributeError),
        "__name__": f"namedtuple_{typename}",
    }
    exec(s, namespace)
    __new__ = namespace["__new__"]
    __new__.__doc__ = f"Create new instance of {typename}({full_list})"
    __init__ = namespace["__init__"]
    __init__.__doc__ = f"Instantiate instance of {typename}({full_list})"
    __setattr__ = namespace["__setattr__"]

    def __repr__(self):
        "Return a nicely formatted representation string"
        return self.__class__.__name__ + repr_fmt % self._asdict()

    def _asdict(self):
        "Return a new dict which maps field names to their values."
        out = _dict(_zip(self._fields, self))
        out.update(self.__dict__)
        return out

    def __getnewargs_ex__(self):
        "Return self as a plain tuple.  Used by copy and pickle."
        return _tuple(self), self.__dict__

    # Modify function metadata to help with introspection and debugging
    for method in (__new__, __repr__, _asdict, __getnewargs_ex__):
        method.__qualname__ = f"{typename}.{method.__name__}"

    # Build-up the class namespace dictionary
    # and use type() to build the result class
    class_namespace = {
        "__doc__": f"{typename}({full_list})",
        "_fields": field_names,
        "__new__": __new__,
        "__init__": __init__,
        "__repr__": __repr__,
        "__setattr__": __setattr__,
        "_asdict": _asdict,
        "_extra_fields": extra_field_names,
        "__getnewargs_ex__": __getnewargs_ex__,
        # _field_defaults and _replace are added to get Polars to detect
        # a bunch object as a namedtuple. See gh-22450
        "_field_defaults": {},
        "_replace": None,
    }
    for index, name in enumerate(field_names):

        def _get(self, index=index):
            return self[index]

        class_namespace[name] = property(_get)
    for name in extra_field_names:

        def _get(self, name=name):
            return self.__dict__[name]

        class_namespace[name] = property(_get)

    result = type(typename, (tuple,), class_namespace)

    # For pickling to work, the __module__ variable needs to be set to the
    # frame where the named tuple is created.  Bypass this step in environments
    # where sys._getframe is not defined (Jython for example) or sys._getframe
    # is not defined for arguments greater than 0 (IronPython), or where the
    # user has specified a particular module.
    if module is None:
        try:
            module = _sys._getframe(1).f_globals.get("__name__", "__main__")
        except (AttributeError, ValueError):
            pass
    if module is not None:
        result.__module__ = module
        __new__.__module__ = module

    return result


TtestResultBase = _make_tuple_bunch("TtestResultBase", ["statistic", "pvalue"], ["df"])


class TtestResult(TtestResultBase):
    """
    Result of a t-test.

    Based on https://github.com/scipy/scipy/blob/e407bc4d6ee71ec1a23fce9d95b58e28451e4d94/scipy/stats/_stats_py.py#L5985
    See the documentation of the particular t-test function for more
    information about the definition of the statistic and meaning of
    the confidence interval.

    Attributes
    ----------
    statistic : float or array
        The t-statistic of the sample.
    pvalue : float or array
        The p-value associated with the given alternative.
    df : float or array
        The number of degrees of freedom used in calculation of the
        t-statistic; this is one less than the size of the sample
        (``a.shape[axis]-1`` if there are no masked elements or omitted NaNs).

    Methods
    -------
    confidence_interval
        Computes a confidence interval around the population statistic
        for the given confidence level.
        The confidence interval is returned in a ``namedtuple`` with
        fields `low` and `high`.

    """

    def __init__(
        self,
        statistic,
        pvalue,
        df,  # public
        alternative,
        standard_error,
        estimate,  # private
        statistic_np=None,
    ):
        super().__init__(statistic, pvalue, df=df)
        # private
        self._alternative = alternative
        self._standard_error = standard_error  # denominator of t-statistic
        self._estimate = estimate  # point estimate of sample mean
        self._statistic_np = statistic if statistic_np is None else statistic_np
        self._dtype = statistic.dtype

    def confidence_interval(self, confidence_level=0.95):
        """
        Parameters
        ----------
        confidence_level : float
            The confidence level for the calculation of the population mean
            confidence interval. Default is 0.95.

        Returns
        -------
        ci : namedtuple
            The confidence interval is returned in a ``namedtuple`` with
            fields `low` and `high`.

        """
        low, high = _t_confidence_interval(
            self.df,
            self._statistic_np,
            confidence_level,
            self._alternative,
            self._dtype,
        )
        low = low * self._standard_error + self._estimate
        high = high * self._standard_error + self._estimate
        return ConfidenceInterval(low=low, high=high)


def _chk_asarray(a, axis):
    """
    Based on https://github.com/scipy/scipy/blob/e407bc4d6ee71ec1a23fce9d95b58e28451e4d94/scipy/stats/_stats_py.py#L113
    """
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis

    if a.ndim == 0:
        a = np.atleast_1d(a)

    return a, outaxis


def _get_pvalue(statistic, distribution, alternative, symmetric=True):
    """
    Get p-value given the statistic, (continuous) distribution, and alternative
    Based on https://github.com/scipy/scipy/blob/e407bc4d6ee71ec1a23fce9d95b58e28451e4d94/scipy/stats/_stats_py.py#L1571
    """
    if alternative == "less":
        pvalue = distribution.cdf(statistic)
    elif alternative == "greater":
        pvalue = distribution.sf(statistic)
    elif alternative == "two-sided":
        pvalue = 2 * (
            distribution.sf(np.abs(statistic))
            if symmetric
            else np.minimum(distribution.cdf(statistic), distribution.sf(statistic))
        )
    else:
        message = "`alternative` must be 'less', 'greater', or 'two-sided'."
        raise ValueError(message)

    return pvalue


def nadeau_bengio_ttest(
    a,
    popmean,
    test_frac,
    axis=0,
    nan_policy="propagate",
    alternative="greater",
):
    """
    One-sample t-test with variance corrected using Nadeau & Bengio.

    Simplification of https://github.com/scipy/scipy/blob/b1296b9b4393e251511fe8fdd3e58c22a1124899/scipy/stats/_stats_py.py#L6035-L6233
    Only support numpy backend and don't support masked data

    This is a modification of scipy.stats.ttest_1samp that applies the
    Nadeau & Bengio correction :footcite::`nadeau1999inference` to the variance
    estimate to account for dependence between repeated cross-validation estimates.

    Parameters
    ----------
    a : array_like
        Sample data. The axis specified by `axis` is the sample axis.
    popmean : scalar or array_like
        Expected value in null hypothesis. If array_like, must be
        broadcastable to the shape of the mean of `a` along `axis`.
    test_frac : float
        Fraction of the data used for testing (test set size / total
        samples). Used by the Nadeau & Bengio correction
        :footcite::`nadeau1999inference` when adjusting the sample variance.
    axis : int or None, optional
        Axis along which to compute the test. Default is 0. If None, the
        input array is flattened.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains NaNs. Default is
        'propagate'. Note: nan handling is performed by the
        `_axis_nan_policy` decorator; the parameter remains in the
        signature for compatibility.
    alternative : {'two-sided', 'greater', 'less'}, optional
        Defines the alternative hypothesis. Default is 'greater'.

    Returns
    -------
    TtestResult
        Named tuple with fields: statistic, pvalue, df, alternative,
        standard_error, estimate. The `statistic` and `pvalue` have the
        Nadeau & Bengio corrected standard error applied.

    Notes
    -----
    The variance is corrected using the factor proposed by Nadeau & Bengio :footcite::`nadeau1999inference`
    to account for dependence across repeated evaluations:
    corrected_var = var * (1/kr + test_frac),
    where kr is the number of repeated evaluations along `axis`.

    This function preserves the interface of scipy.stats.ttest_1samp while
    replacing the usual sample variance by the corrected variance.

    References
    ----------
    .. footbibliography::
    """
    a, axis = _chk_asarray(a, axis)

    n = a.shape[axis]
    df = n - 1

    if a.shape[axis] == 0:
        # This is really only needed for *testing* _axis_nan_policy decorator
        # It won't happen when the decorator is used.
        NaN = np.full((), np.nan, dtype=float)
        return TtestResult(
            NaN, NaN, df=NaN, alternative=NaN, standard_error=NaN, estimate=NaN
        )

    mean = np.mean(a, axis=axis)
    try:
        popmean = np.asarray(popmean)
        popmean = np.squeeze(popmean, axis=axis) if popmean.ndim > 0 else popmean
    except ValueError as e:
        raise ValueError("`popmean.shape[axis]` must equal 1.") from e
    d = mean - popmean
    v = _var(a, axis=axis, ddof=1)
    ######### ADD correction of ttest
    denom = np.sqrt(v * (1 / n + test_frac))
    ################################

    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.divide(d, denom)
        t = t[()] if t.ndim == 0 else t

    dist = _SimpleStudentT(np.asarray(df, dtype=t.dtype))
    prob = _get_pvalue(t, dist, alternative)
    prob = prob[()] if prob.ndim == 0 else prob

    # when nan_policy='omit', `df` can be different for different axis-slices
    df = np.broadcast_to(np.asarray(df), t.shape)
    df = df[()] if df.ndim == 0 else df
    # _axis_nan_policy decorator doesn't play well with strings
    alternative_num = {"less": -1, "two-sided": 0, "greater": 1}[alternative]
    return TtestResult(
        t,
        prob,
        df=df,
        alternative=alternative_num,
        standard_error=denom,
        estimate=mean,
        statistic_np=np.asarray(t),
    )
