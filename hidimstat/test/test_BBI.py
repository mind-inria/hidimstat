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


def _generate_data(n_samples=100, n_features=10, prob_type="regression",
                   grps_exp=False, seed=2024):

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
    X["Val1"] = rng.choice(["Car", "Moto", "Velo", "Plane", "Boat"],
                           size=n_samples)
    X["Val2"] = rng.choice(["Car_1", "Moto_1", "Velo_1", "Plane_1"],
                           size=n_samples)
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
    # ## Providing or not a DNN learner as a predictor for the second block
    # ## or the default Random Forest will be used
    # new_second_predictor = False

    # if new_second_predictor:
    #     imp_pred = {
    #         "classification": DNN_learner(prob_type="classification", encode=True),
    #         "regression": DNN_learner(prob_type="regression"),
    #         "ordinal": DNN_learner(prob_type="classification", encode=True),
    #     }
    # else:
    #     imp_pred = None

    # if k_fold == 0:
    #     train_index = rng.choice(
    #         X.shape[0], size=int(X.shape[0] * (1 - 0.2)), replace=False
    #     )
    #     test_index = np.array([ind for ind in range(X.shape[0]) if ind not in train_index])
    #     X_train, X_test = X.loc[train_index, :], X.loc[test_index, :]
    #     y_train, y_test = y[train_index], y[test_index]
    # else:
    #     X_train = X.copy()
    #     y_train = y.copy()
    #     X_test = X.copy()
    #     y_test = y.copy()

    # if prob_type == "regression":
    #     rf = RandomForestRegressor(random_state=2023)
    # else:
    #     rf = RandomForestClassifier(random_state=2023)
    # dict_hyper = {"max_depth": [2, 5, 10, 20]}
    return X, y, grps, list_nominal


def test_BBI_DNN():

    X, y, grps, list_nominal = _generate_data()
    # Permutation Method
    bbi_perm = BlockBasedImportance(
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
    bbi_perm.fit(X, y)
    results_perm = bbi_perm.compute_importance()
    assert len(results_perm["pval"]) == X.shape[1]


# def test_BBI_samplingRF():

#     X, y, grps, list_nominal = _generate_data()
#     # Permutation Method
#     bbi_perm = BlockBasedImportance(
#         estimator=None,
#         importance_estimator="Mod_RF",
#         do_hyper=True,
#         dict_hyper=None,
#         conditional=False,
#         group_stacking=False,
#         prob_type="regression",
#         k_fold=2,
#         list_nominal=list_nominal,
#         n_jobs=10,
#         verbose=0,
#         n_perm=100,
#         )
#     bbi_perm.fit(X, y)
#     results_perm = bbi_perm.compute_importance()
#     assert len(results_perm["pval"]) == X.shape[1]
    # # Conditional Method
    # bbi_cond = BlockBasedImportance(
    #     estimator='RF',
    #     importance_estimator="Mod_RF",
    #     do_hyper=True,
    #     dict_hyper=None,
    #     conditional=True,
    #     group_stacking=False,
    #     prob_type="regression",
    #     k_fold=k_fold,
    #     list_nominal=list_nominal,
    #     n_jobs=10,
    #     verbose=0,
    #     n_perm=100,
    #     )
    # bbi_cond.fit(X, y)
    # results_cond = bbi_cond.compute_importance()
    # pvals_cond = -np.log10(results_cond["pval"] + 1e-5)
    # assert len(pvals_cond) == X.shape[1]


# def test_BBI_residuals():
#     # Permutation Method
#     bbi_perm = BlockBasedImportance(
#         estimator='RF',
#         importance_estimator=None,
#         do_hyper=True,
#         dict_hyper=None,
#         conditional=False,
#         group_stacking=False,
#         prob_type="regression",
#         k_fold=k_fold,
#         list_nominal=list_nominal,
#         n_jobs=10,
#         verbose=0,
#         n_perm=100,
#         )
#     bbi_perm.fit(X, y)
#     results_perm = bbi_perm.compute_importance()
#     pvals_perm = -np.log10(results_perm["pval"] + 1e-10)
#     assert len(pvals_perm) == X.shape[1]

#     # Conditional Method
#     bbi_cond = BlockBasedImportance(
#         estimator='RF',
#         importance_estimator=None,
#         do_hyper=True,
#         dict_hyper=None,
#         conditional=True,
#         group_stacking=False,
#         prob_type="regression",
#         k_fold=k_fold,
#         list_nominal=list_nominal,
#         n_jobs=10,
#         verbose=0,
#         n_perm=100,
#         )
#     bbi_cond.fit(X, y)
#     results_cond = bbi_cond.compute_importance()
#     pvals_cond = -np.log10(results_cond["pval"] + 1e-5)
#     assert len(pvals_cond) == X.shape[1]
