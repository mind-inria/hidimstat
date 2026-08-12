from collections import namedtuple

TtestResult = namedtuple("TtestResult", ["statistic", "pvalue"])
TtestResult.__doc__ = "Results from a statistical test."
