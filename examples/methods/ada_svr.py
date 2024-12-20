"""
ADA-SVR: Adaptive Permutation Threshold Support Vector Regression
==================================================================
Statistical inference procedure presented in :footcite:t:`gaonkar_deriving_2012`.
"""

#############################################################################
# Imports needed for this script
# ------------------------------
import matplotlib.pyplot as plt
import numpy as np
from hidimstat.ada_svr import ada_svr
from hidimstat.scenario import multivariate_1D_simulation
from examples._utils.plot_dataset import plot_dataset1D

#############################################################################
# Generate toy dataset
# ---------------------------
#

# Parameters for the generation of data
n_samples, n_features = 20, 50
support_size = 4

X, y, beta, _ = multivariate_1D_simulation(n_samples=5, n_features=20,
                                             support_size=1, sigma=0.1,
                                             shuffle=False, seed=42)
plot_dataset1D(X=X, y=y, beta=beta)

plt.figure()
plt.plot(np.arange(n_features))
plt.xlabel("Features")
plt.ylabel("Values")
plt.show()

#############################################################################
# Usage the methods
# -----------------
#
# The method is called as follows:
#
# .. code-block:: python
#
#     ada_svr(X, y, rcond=1e-3)
#
# where:
#
# - ``X`` is the design matrix of shape ``(n_samples, n_features)``.
#
# - ``y`` is the target vector of shape ``(n_samples,)``.
#
# - ``rcond`` is the cuoff for small singular values ``float``.
#
# The method returns a dictionary containing the following keys:
#
# - ``beta_hat`` is the estimated of vector of importances variable of shape ``(n_features,)``
#
# - ``scale`` is the vector of the standard deviation of a gaussians distribution of beta_hat ``(n_features,)``



#############################################################################
# References
# ----------
# .. footbibliography::