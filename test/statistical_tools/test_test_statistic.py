import numpy as np
import pytest
from scipy.stats import ttest_1samp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from hidimstat import CFI
import warnings

import numpy as np
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

from hidimstat.statistical_tools.test_statistic import nadeau_bengio_ttest


def check_named_results(res, attributes, ma=False, xp=None):
    """
    Copy from https://github.com/scipy/scipy/blob/2878daa7083375847e3a181553b146e843efcfad/scipy/stats/tests/common_tests.py#L17
    """
    for i, attr in enumerate(attributes):
        if ma:
            ma_npt.assert_equal(res[i], getattr(res, attr))
        else:
            npt.assert_equal(res[i], getattr(res, attr))


@pytest.mark.parametrize(
    "n_samples, n_features, support_size, rho, seed, value, signal_noise_ratio, rho_serial",
    [[100, 6, 2, 0.2, 0, 1.0, 10.0, 0.0]],
)
def test_ttest_1samp_corrected_NB(data_generator):
    """
    Test the corrected one-sample t-test (Nadeau & Bengio) implementation in a
    cross-validation setting with a linear synthetic dataset.
     - Test that the test statistic is computed on the correct axis (over folds).
     - Compare p-values with and without correction. The corrected p-values should be
       larger (more conservative).
     - Check that it allows to identify important features.
    """
    X, y, important_features, _ = data_generator
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    importance_list = []
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = LinearRegression().fit(X_train, y_train)
        vim = CFI(
            estimator=model,
            imputation_model_continuous=LinearRegression(),
            random_state=0,
        )
        vim.fit(X_train, y_train)
        importances = vim.importance(X_test, y_test)
        importance_list.append(importances["importance"])
    importance_array = np.array(importance_list)

    pvalue_corr = nadeau_bengio_ttest(importance_array, 0, test_frac=0.2).pvalue
    pvalue = ttest_1samp(importance_array, 0, alternative="greater").pvalue
    n_features = X.shape[1]
    alpha = 0.05
    assert pvalue_corr.shape == (n_features,)
    assert np.all(pvalue_corr >= pvalue)
    assert np.all(pvalue_corr[important_features] < alpha)
    assert np.all(
        pvalue_corr[np.setdiff1d(np.arange(n_features), important_features)] >= alpha
    )


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
        res1 = nadeau_bengio_ttest(
            outcome[:, 0],
            1,
            test_frac=1 - 1 / outcome[:, 0].shape[0],
            alternative="two-sided",
        )
        res2 = mstats.ttest_1samp(outcome[:, 0], 1)
        assert_allclose(res1, res2)
        res3 = nadeau_bengio_ttest(
            outcome[:, 0], 1, test_frac=0.1, alternative="two-sided"
        )
        assert not np.array_equal(res3, res1)
        assert not np.array_equal(res3, res2)

    def test_fully_masked(self):
        """Test comparison with fully masked data"""
        # outcome = ma.masked_array(self.rng.standard_normal(3), mask=[1, 1, 1])
        expected = (np.nan, np.nan)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "invalid value encountered in absolute", RuntimeWarning
            )
            for pair in [((np.nan, np.nan), 0.0, 0.1)]:
                t, p = nadeau_bengio_ttest(*pair)
                assert_array_equal(p, expected)
                assert_array_equal(t, expected)

    def test_result_attributes(self):
        """Test attribute"""
        outcome = self.rng.standard_normal((20, 4)) + [0, 0, 1, 2]

        res = nadeau_bengio_ttest(outcome[:, 0], 1, 0.1)
        attributes = ("statistic", "pvalue")
        check_named_results(res, attributes, ma=True)

    def test_empty(self):
        """Test for empty data"""
        res1 = nadeau_bengio_ttest([], 1, 0.1)
        assert_(np.all(np.isnan(res1)))

    def test_zero_division(self):
        """Test for zero division"""
        t, p = nadeau_bengio_ttest([0, 0, 0], 1, 0.1, alternative="two-sided")
        assert_equal((np.abs(t), p), (np.inf, 0))

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "invalid value encountered in absolute", RuntimeWarning
            )
            t, p = nadeau_bengio_ttest([0, 0, 0], 0, 0.1)
            assert_(np.isnan(t))
            assert_array_equal(p, (np.nan, np.nan))

    @pytest.mark.parametrize("alternative", ["less", "greater"])
    def test_alternative(self, alternative):
        """Test option alternative"""
        x = stats.norm.rvs(loc=10, scale=2, size=100, random_state=123)

        t_ex, p_ex = nadeau_bengio_ttest(
            x, 9, test_frac=1 - 1 / x.shape[0], alternative=alternative
        )
        t, p = mstats.ttest_1samp(x, 9, alternative=alternative)
        assert_allclose(t, t_ex, rtol=1e-14)
        assert_allclose(p, p_ex, rtol=1e-14)

        t_ex_1, p_ex_1 = nadeau_bengio_ttest(
            x, 9, test_frac=0.1, alternative=alternative
        )
        assert not np.array_equal(t_ex_1, t_ex)
        assert not np.array_equal(t_ex_1, t)
        assert not np.array_equal(p_ex_1, p_ex)
        assert not np.array_equal(p_ex_1, p)

        # test with masked arrays
        x[1:10] = np.nan
        x = np.ma.masked_array(x, mask=np.isnan(x))
        t_ex, p_ex = nadeau_bengio_ttest(
            x.compressed(),
            9,
            test_frac=1 - 1 / (x.shape[0] - 9),
            alternative=alternative,
        )
        t, p = mstats.ttest_1samp(x, 9, alternative=alternative)
        assert_allclose(t, t_ex, rtol=1e-14)
        assert_allclose(p, p_ex, rtol=1e-14)

        t_ex_1, p_ex_1 = nadeau_bengio_ttest(
            x, 9, test_frac=0.1, alternative=alternative
        )
        assert not np.array_equal(t_ex_1, t_ex)
        assert not np.array_equal(t_ex_1, t)
        assert not np.array_equal(p_ex_1, p_ex)
        assert not np.array_equal(p_ex_1, p)
