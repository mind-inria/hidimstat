"""
Test the BBI module
"""
import numpy as np
from hidimstat.BBI import BlockBasedImportance
from sklearn.datasets import load_diabetes
# Fixing the random seed
rng = np.random.RandomState(2024)

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Use or not a cross-validation with the provided learner
k_fold = 2
# Identifying the categorical (nominal & ordinal) variables
list_nominal = {}

def test_BBI():
    # Permutation Method
    bbi_perm = BlockBasedImportance(
        estimator='RF',
        importance_estimator="Mod_RF",
        do_hyper=True,
        dict_hyper=None,
        conditional=False,
        group_stacking=False,
        prob_type="regression",
        k_fold=k_fold,
        list_nominal=list_nominal,
        n_jobs=10,
        verbose=0,
        n_perm=100,
        )
    bbi_perm.fit(X, y)
    results_perm = bbi_perm.compute_importance()
    pvals_perm = -np.log10(results_perm["pval"] + 1e-10)
    assert len(pvals_perm) == X.shape[1]

    # Conditional Method
    bbi_cond = BlockBasedImportance(
        estimator='RF',
        importance_estimator="Mod_RF",
        do_hyper=True,
        dict_hyper=None,
        conditional=True,
        group_stacking=False,
        prob_type="regression",
        k_fold=k_fold,
        list_nominal=list_nominal,
        n_jobs=10,
        verbose=0,
        n_perm=100,
        )
    bbi_cond.fit(X, y)
    results_cond = bbi_cond.compute_importance()
    pvals_cond = -np.log10(results_cond["pval"] + 1e-5)
    assert len(pvals_cond) == X.shape[1]