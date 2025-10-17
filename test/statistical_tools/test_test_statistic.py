import numpy as np
import pytest
from scipy.stats import ttest_1samp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from hidimstat import CFI
from hidimstat.statistical_tools import ttest_1samp_corrected_NB


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

    pvalue_corr = ttest_1samp_corrected_NB(importance_array, 0, test_frac=0.2).pvalue
    pvalue = ttest_1samp(importance_array, 0, alternative="greater").pvalue
    n_features = X.shape[1]
    alpha = 0.05
    assert pvalue_corr.shape == (n_features,)
    assert np.all(pvalue_corr >= pvalue)
    assert np.all(pvalue_corr[important_features] < alpha)
    assert np.all(
        pvalue_corr[np.setdiff1d(np.arange(n_features), important_features)] >= alpha
    )
