"""
Different API
=============

example of API
"""

import numpy as np
from scipy.stats import ttest_1samp
from sklearn.datasets import make_circles
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
#############################################################################
# Generate data where classes are not linearly separable
# --------------------------------------------------------------
rng = np.random.RandomState(0)
X, y = make_circles(n_samples=500, noise=0.1, factor=0.6, random_state=rng)

#############################################################################
# 4:API O2/07 
# ================
from hidimstat_mock.v4 import PFI, LOCO, D0CRT, Knockoff
cv = KFold(n_splits=5, shuffle=True, random_state=0)
###############################################################################
# Variable Importance:
# --------------------
# Variable importance and selection using Permutation Features Importance

vi = PFI(estimator=LogisticRegressionCV(Cs=np.logspace(-3, 3, 5)).fit(X, y))
vi.fit(X, y)
importances_vi = vi.importance(None, None) # vector of n_features 
importances_vi, None = vi.importance_fit(X, y, cv) # vector of n_features, not p-value 
selection_vi = vi.selection(percentage=0.1, threshold=None, k_best=None, threshold_pvalue=None) # binary vector of n_features
###############################################################################
# Controlled Variable Importance:
# -------------------------------
# Variable importance, p-value and selection using LOCO
cvi = LOCO(estimator=LogisticRegressionCV(Cs=np.logspace(-3, 3, 5)).fit(X, y))
cvi.fit(X[train], y[train]) 
importances_cvi = cvi.importance(X[test], y[test]) # vector of n_features 
importances_cvi, pvalue_cvi = cvi.importance(X, y, cv) # vector of n_features 
selection_cvi = cvi.selection(percentage=None, threshold=2.0, k_best=None, threshold_pvalue=None) # binary vector of n_features
###############################################################################
# Variable Selection:
# ------------------------------
# P-values and selection using d0CRT
vs = D0CRT()
vs.fit(X, y)
# In this case importance and p-value can be very similar
importances_vs = vs.importance(None, None) # vector of n_features 
importance_vs, pvalue_vs = vs.importance(X, y, None) # vector of n_features 
selection_vs = vs.selection(percentage=None, threshold=None, k_best=20.0, threshold_pvalue=None) # binary vector of n_features
# ----------------------------
# selection using Knockoff
vs = Knockoff()
vs.fit(X, y)
importance_vs = vs.importance(None, None) # vector of n_features 
importance_vs, pvalue_vs = vs.importance(X, y, None) # vector of n_features 
selection_vs = vs.selection(percentage=None, threshold=None, k_best=20.0, threshold_pvalue=0.8) # binary vector of n_features

#############################################################################
# 3:API 11/06 
# ================
from hidimstat_mock.v3 import PFI, LOCO, D0CRT, Knockoff
cv = KFold(n_splits=5, shuffle=True, random_state=0)
###############################################################################
# Variable Importance:
# --------------------
# Variable importance and selection using Permutation Features Importance

vi = PFI(estimator=LogisticRegressionCV(Cs=np.logspace(-3, 3, 5)).fit(X, y))
vi.fit(X, y)
importances_vi = vi.importance(None, None) # vector of n_features 
importances_vi, None = vi.importance_fit(X, y, cv) # vector of n_features, not p-value 
selection_vi = selection(importances_vi, None, percentage=0.1) # binary vector of n_features
###############################################################################
# Controlled Variable Importance:
# -------------------------------
# Variable importance, p-value and selection using LOCO
cvi = LOCO(estimator=LogisticRegressionCV(Cs=np.logspace(-3, 3, 5)).fit(X, y))
cvi.fit(X[train], y[train]) 
importances_cvi = cvi.importance(X[test], y[test]) # vector of n_features 
importances_cvi, pvalue_cvi = cvi.importance(X, y, cv) # vector of n_features 
selection_cvi = selection(importances_cvi, pvalue_cvi, threshold=2.0) # binary vector of n_features
###############################################################################
# Individual Variable Selection:
# ------------------------------
# P-values and selection using d0CRT
ivs = D0CRT()
ivs.fit(X, y)
# In this case importance and p-value can be very similar
importances_ivs = ivs.importance(None, None) # vector of n_features 
importance_ivs, pvalue_ivs = ivs.importance(X, y, None) # vector of n_features 
selection_ivs = selection(importance_ivs, pvalue_ivs, k_best=20.0) # binary vector of n_features
###############################################################################
# Multiple Variable Selection:
# ----------------------------
# selection using Knockoff
mvs = Knockoff()
mvs.fit(X, y)
importance_mvs = mvs.importance(None, None) # vector of n_features 
importance_mvs, pvalue_mvs = mvs.importance(X, y, None) # vector of n_features 
selection_mvs = selection(importance_mvs, pvalue_mvs, k_best=20.0, threshold_pvalue=0.8) # binary vector of n_features

#############################################################################
# 2:API 26 Mai 
# ================
from hidimstat_mock.v2 import PFI, LOCO, D0CRT, Knockoff
###############################################################################
# Variable Importance:
# --------------------
# Variable importance and selection using Permutation Features Importance

vi = PFI(estimator=LogisticRegressionCV(Cs=np.logspace(-3, 3, 5)).fit(X, y))
vi.fit(X, y)
importances_vi = vi.importance(None, None) # vector of n_features 
# REMOVED: selection_vi = vi.selection(10) # binary vector of n_features
###############################################################################
# Controlled Variable Importance:
# -------------------------------
# Variable importance, p-value and selection using LOCO
cv = KFold(n_splits=5, shuffle=True, random_state=0)
importances_cvi = []
for train, test in cv.split(X):
    cvi = LOCO(
        estimator=LogisticRegressionCV(Cs=np.logspace(-3, 3, 5)).fit(X[train], y[train])
        )
    cvi.fit(X[train], y[train]) 
    importances = cvi.importance(X[test], y[test]) # vector of n_features 
    importances_cvi.append(importances)
selection, pvalue = cvi.selection(X, y, 10, cv=cv) # binary vector of n_features
###############################################################################
# Individual Variable Selection:
# ------------------------------
# P-values and selection using d0CRT
ivs = D0CRT()
ivs.fit(X, y)
# REMOVED: pvalue_ivs = ivs.importance(None, None) # vector of n_features 
selection_vi, pvalue_ivs = ivs.selection(None, None, 0.1) # binary vector of n_features
###############################################################################
# Multiple Variable Selection:
# ----------------------------
# selection using Knockoff
mvs = Knockoff()
mvs.fit(X, y)
# REMOVED: score_mvs = mvs.importance(None, None) # vector of n_features 
selection_mvs, _ = mvs.selection(None, None, FDP=0.1) # ??2 output?? binary vector of n_features
# WARNING: knoclkoff doesn't have p-values?

###############################################################################
# 1:API post sprint  4/4/2025
# ===========================
from hidimstat_mock.v1 import PFI, LOCO, D0CRT, Knockoff
###############################################################################
# Variable Importance:
# --------------------
# Variable importance and selection using Permutation Features Importance

vi = PFI(estimator=LogisticRegressionCV(Cs=np.logspace(-3, 3, 5)).fit(X, y))
vi.fit(X, y)
importances_vi = vi.importance() # vector of n_features 
selection_vi = vi.selection(10) # binary vector of n_features
###############################################################################
# Controlled Variable Importance:
# -------------------------------
# Variable importance, p-value and selection using LOCO
cv = KFold(n_splits=5, shuffle=True, random_state=0)
importances_cvi = []
selection_cvi = []
for train, test in cv.split(X):
    cvi = LOCO(
        estimator=LogisticRegressionCV(Cs=np.logspace(-3, 3, 5)).fit(X[train], y[train])
        )
    cvi.fit(X[train], y[train]) 
    importances = cvi.importance(X[test], y[test]) # vector of n_features 
    importances_cvi.append(importances)
    selection = cvi.selection(10) # binary vector of n_features
    selection_cvi.append(selection)
_, pval_cvi = ttest_1samp(importances_cvi, 0, axis=0, alternative="greater")
###############################################################################
# Individual Variable Selection:
# ------------------------------
# P-values and selection using d0CRT
ivs = D0CRT()
ivs.fit(X, y)
pvalue_ivs = ivs.importance() # vector of n_features 
selection_vi = ivs.selection(0.1) # binary vector of n_features
###############################################################################
# Multiple Variable Selection:
# ----------------------------
# selection using Knockoff
mvs = Knockoff()
mvs.fit(X, y)
score_mvs = mvs.importance() # vector of n_features 
selection_mvs = mvs.selection(FDP=0.1) # binary vector of n_features