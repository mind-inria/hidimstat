from collections import namedtuple

TestResult = namedtuple("TestResult", ["statistic", "pvalue"])
TestResult.__doc__ = "Results from a statistical test."
