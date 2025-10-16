import warnings

import numpy as np
import numpy.ma as ma
import numpy.ma.testutils as ma_npt
import numpy.testing as npt
import pytest
import scipy.stats.mstats as mstats
from numpy.ma.testutils import (
    assert_,
    assert_allclose,
    assert_array_equal,
    assert_equal,
)
from scipy import stats

from hidimstat.statistical_tools.test_statistic import ttest_1samp_corrected_NB


def check_named_results(res, attributes, ma=False, xp=None):
    """
    Copy from https://github.com/scipy/scipy/blob/2878daa7083375847e3a181553b146e843efcfad/scipy/stats/tests/common_tests.py#L17
    """
    for i, attr in enumerate(attributes):
        if ma:
            ma_npt.assert_equal(res[i], getattr(res, attr))
        else:
            npt.assert_equal(res[i], getattr(res, attr))


class TestTtest_1samp:
    """
    Copy from https://github.com/scipy/scipy/blob/2878daa7083375847e3a181553b146e843efcfad/scipy/stats/tests/test_mstats_basic.py#L1455
    """

    def setup_method(self):
        self.rng = np.random.default_rng(6043813830)

    def test_vs_nonmasked(self):
        """Test comparison with masked version"""
        outcome = self.rng.standard_normal((20, 4)) + [0, 0, 1, 2]

        # 1-D inputs
        res1 = ttest_1samp_corrected_NB(
            outcome[:, 0],
            1,
            test_frac=1 - 1 / outcome[:, 0].shape[0],
            alternative="two-sided",
        )
        res2 = mstats.ttest_1samp(outcome[:, 0], 1)
        assert_allclose(res1, res2)
        res3 = ttest_1samp_corrected_NB(
            outcome[:, 0], 1, test_frac=0.1, alternative="two-sided"
        )
        assert not np.array_equal(res3, res1)
        assert not np.array_equal(res3, res2)

    def test_fully_masked(self):
        """Test comparison with fully masked data"""
        outcome = ma.masked_array(self.rng.standard_normal(3), mask=[1, 1, 1])
        expected = (np.nan, np.nan)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "invalid value encountered in absolute", RuntimeWarning
            )
            for pair in [((np.nan, np.nan), 0.0, 0.1), (outcome, 0.0, 0.1)]:
                t, p = ttest_1samp_corrected_NB(*pair)
                assert_array_equal(p, expected)
                assert_array_equal(t, expected)

    def test_result_attributes(self):
        """Test attribute"""
        outcome = self.rng.standard_normal((20, 4)) + [0, 0, 1, 2]

        res = ttest_1samp_corrected_NB(outcome[:, 0], 1, 0.1)
        attributes = ("statistic", "pvalue")
        check_named_results(res, attributes, ma=True)

    def test_empty(self):
        """Test for empty data"""
        res1 = ttest_1samp_corrected_NB([], 1, 0.1)
        assert_(np.all(np.isnan(res1)))

    def test_zero_division(self):
        """Test for zero division"""
        t, p = ttest_1samp_corrected_NB([0, 0, 0], 1, 0.1, alternative="two-sided")
        assert_equal((np.abs(t), p), (np.inf, 0))

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "invalid value encountered in absolute", RuntimeWarning
            )
            t, p = ttest_1samp_corrected_NB([0, 0, 0], 0, 0.1)
            assert_(np.isnan(t))
            assert_array_equal(p, (np.nan, np.nan))

    @pytest.mark.parametrize("alternative", ["less", "greater"])
    def test_alternative(self, alternative):
        """Test option alternative"""
        x = stats.norm.rvs(loc=10, scale=2, size=100, random_state=123)

        t_ex, p_ex = ttest_1samp_corrected_NB(
            x, 9, test_frac=1 - 1 / x.shape[0], alternative=alternative
        )
        t, p = mstats.ttest_1samp(x, 9, alternative=alternative)
        assert_allclose(t, t_ex, rtol=1e-14)
        assert_allclose(p, p_ex, rtol=1e-14)

        t_ex_1, p_ex_1 = ttest_1samp_corrected_NB(
            x, 9, test_frac=0.1, alternative=alternative
        )
        assert not np.array_equal(t_ex_1, t_ex)
        assert not np.array_equal(t_ex_1, t)
        assert not np.array_equal(p_ex_1, p_ex)
        assert not np.array_equal(p_ex_1, p)

        # test with masked arrays
        x[1:10] = np.nan
        x = np.ma.masked_array(x, mask=np.isnan(x))
        t_ex, p_ex = ttest_1samp_corrected_NB(
            x.compressed(),
            9,
            test_frac=1 - 1 / (x.shape[0] - 9),
            alternative=alternative,
        )
        t, p = mstats.ttest_1samp(x, 9, alternative=alternative)
        assert_allclose(t, t_ex, rtol=1e-14)
        assert_allclose(p, p_ex, rtol=1e-14)

        t_ex_1, p_ex_1 = ttest_1samp_corrected_NB(
            x, 9, test_frac=0.1, alternative=alternative
        )
        assert not np.array_equal(t_ex_1, t_ex)
        assert not np.array_equal(t_ex_1, t)
        assert not np.array_equal(p_ex_1, p_ex)
        assert not np.array_equal(p_ex_1, p)
