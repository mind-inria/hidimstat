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

###############################################################################
# 1:API post sprint
# ==================
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
selection_mvs, ?_? = mvs.selection(None, None, FDP=0.1) # binary vector of n_features
# WARNING: knoclkoff doesn't have p-values?