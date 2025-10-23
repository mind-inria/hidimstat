"""
Test the BBI module
"""

import numpy as np
import pandas as pd
from hidimstat.BBI import BlockBasedImportance
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Fixing the random seed
rng = np.random.RandomState(2024)


def _generate_data(
    n_samples=100, n_features=10, prob_type="regression", grps_exp=False, seed=2024
):

    if prob_type == "regression":
        X, y = make_regression(
            n_samples=n_samples,
            noise=0.2,
            n_features=n_features,
            random_state=seed,
        )
    else:
        X, y = make_classification(
            n_samples=n_samples,
            n_classes=2,
            n_informative=5,
            n_features=n_features,
            random_state=seed,
        )
        y = np.array([str(i) for i in y])

    X = pd.DataFrame(X, columns=[f"col{i+1}" for i in range(n_features)])
    # Nominal variables
    X["Val1"] = rng.choice(["Car", "Moto", "Velo", "Plane", "Boat"], size=n_samples)
    X["Val2"] = rng.choice(["Car_1", "Moto_1", "Velo_1", "Plane_1"], size=n_samples)
    # Ordinal variables
    X["Val3"] = rng.choice(np.arange(1, 7), size=n_samples)
    X["Val4"] = rng.choice(np.arange(10), size=n_samples)

    if grps_exp:
        grps = {
            "A0": ["Val1", "Val2"],
            "A": ["col6", "col7", "Val1", "Val2"],
            "B": ["col7", "col8", "col9", "col10", "Val3", "Val4"],
            "C": ["col1", "col2", "col3", "col4"],
            "D": ["col5"],
            "E": ["Val1"],
            "F": ["Val3"],
        }
    else:
        grps = None

    list_nominal = {"nominal": ["Val1", "Val2"], "ordinal": ["Val3", "Val4"]}
    return X, y, grps, list_nominal


def test_BBI_reg():

    X, y, _, list_nominal = _generate_data(prob_type="regression")
    # DNN
    bbi_reg_dnn = BlockBasedImportance(
        estimator=None,
        importance_estimator="Mod_RF",
        do_hyper=True,
        dict_hyper=None,
        conditional=False,
        group_stacking=False,
        prob_type="regression",
        k_fold=2,
        list_nominal=list_nominal,
        n_jobs=10,
        verbose=0,
        n_perm=100,
    )
    bbi_reg_dnn.fit(X, y)
    results_reg_dnn = bbi_reg_dnn.compute_importance()
    assert len(results_reg_dnn["pval"]) == X.shape[1]

    # RF
    bbi_reg_rf = BlockBasedImportance(
        estimator="RF",
        importance_estimator="Mod_RF",
        do_hyper=True,
        dict_hyper=None,
        conditional=False,
        group_stacking=False,
        prob_type="regression",
        k_fold=2,
        list_nominal=list_nominal,
        n_jobs=10,
        verbose=0,
        n_perm=100,
    )
    bbi_reg_rf.fit(X, y)
    results_reg_rf = bbi_reg_rf.compute_importance()
    assert len(results_reg_rf["pval"]) == X.shape[1]


def test_BBI_class():

    X, y, _, list_nominal = _generate_data(prob_type="classification")
    # DNN
    bbi_class_dnn = BlockBasedImportance(
        estimator=None,
        importance_estimator="Mod_RF",
        do_hyper=True,
        dict_hyper=None,
        conditional=False,
        group_stacking=False,
        prob_type="classification",
        k_fold=2,
        list_nominal=list_nominal,
        n_jobs=10,
        verbose=0,
        n_perm=100,
    )
    bbi_class_dnn.fit(X, y)
    results_class_dnn = bbi_class_dnn.compute_importance()
    assert len(results_class_dnn["pval"]) == X.shape[1]

    # RF
    bbi_class_rf = BlockBasedImportance(
        estimator="RF",
        importance_estimator="Mod_RF",
        do_hyper=True,
        dict_hyper=None,
        conditional=False,
        group_stacking=False,
        prob_type="classification",
        k_fold=2,
        list_nominal=list_nominal,
        n_jobs=10,
        verbose=0,
        n_perm=100,
    )
    bbi_class_rf.fit(X, y)
    results_class_rf = bbi_class_rf.compute_importance()
    assert len(results_class_rf["pval"]) == X.shape[1]


def test_BBI_condDNN():

    X, y, _, list_nominal = _generate_data()
    # Compute importance with residuals
    bbi_res = BlockBasedImportance(
        estimator=None,
        importance_estimator=None,
        do_hyper=True,
        dict_hyper=None,
        group_stacking=False,
        prob_type="regression",
        k_fold=2,
        list_nominal=list_nominal,
        n_jobs=10,
        verbose=0,
        n_perm=100,
    )
    bbi_res.fit(X, y)
    results_res = bbi_res.compute_importance()
    assert len(results_res["pval"]) == X.shape[1]

    # Compute importance with sampling RF
    bbi_samp = BlockBasedImportance(
        estimator=None,
        importance_estimator="Mod_RF",
        do_hyper=True,
        dict_hyper=None,
        group_stacking=False,
        prob_type="regression",
        k_fold=2,
        list_nominal=list_nominal,
        n_jobs=10,
        verbose=0,
        n_perm=100,
    )
    bbi_samp.fit(X, y)
    results_samp = bbi_samp.compute_importance()
    assert len(results_samp["pval"]) == X.shape[1]


def test_BBI_permDNN():

    X, y, _, list_nominal = _generate_data()
    bbi_perm = BlockBasedImportance(
        estimator=None,
        importance_estimator="Mod_RF",
        do_hyper=True,
        dict_hyper=None,
        conditional=False,
        group_stacking=False,
        prob_type="regression",
        k_fold=2,
        list_nominal=list_nominal,
        n_jobs=10,
        verbose=0,
        n_perm=100,
    )
    bbi_perm.fit(X, y)
    results_perm = bbi_perm.compute_importance()
    assert len(results_perm["pval"]) == X.shape[1]


def test_BBI_grp():

    X, y, grps, list_nominal = _generate_data(grps_exp=True)
    # No Stacking
    bbi_grp_noStack = BlockBasedImportance(
        estimator="RF",
        importance_estimator="Mod_RF",
        do_hyper=True,
        dict_hyper=None,
        conditional=False,
        groups=grps,
        group_stacking=False,
        prob_type="regression",
        k_fold=2,
        list_nominal=list_nominal,
        n_jobs=10,
        verbose=0,
        n_perm=100,
    )
    bbi_grp_noStack.fit(X, y)
    results_grp_noStack = bbi_grp_noStack.compute_importance()
    assert len(results_grp_noStack["pval"]) == len(grps)

    # Stacking
    bbi_grp_stack = BlockBasedImportance(
        estimator="RF",
        importance_estimator="Mod_RF",
        do_hyper=True,
        dict_hyper=None,
        conditional=False,
        groups=grps,
        group_stacking=True,
        prob_type="regression",
        k_fold=2,
        list_nominal=list_nominal,
        n_jobs=10,
        verbose=0,
        n_perm=100,
    )
    bbi_grp_stack.fit(X, y)
    results_grp_stack = bbi_grp_stack.compute_importance()
    assert len(results_grp_stack["pval"]) == len(grps)
