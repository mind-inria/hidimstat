from .base_perturbation import BasePerturbation
from .clustered_inference import clustered_inference, hd_inference
from .desparsified_lasso import (
    desparsified_lasso,
    desparsified_lasso_pvalue,
    desparsified_group_lasso_pvalue,
)
from .conditional_permutation_importance import CPI
from .desparsified_lasso import desparsified_group_lasso, desparsified_lasso
from .empirical_thresholding import empirical_thresholding
from .ensemble_clustered_inference import ensemble_clustered_inference
from .knockoff_aggregation import knockoff_aggregation
from .knockoffs import model_x_knockoff
from .leave_one_covariate_out import LOCO
from .multi_sample_split import aggregate_quantiles
from .noise_std import reid
from .permutation_importance import PermutationImportance
from .permutation_test import permutation_test, permutation_test_pval
from .scenario import multivariate_1D_simulation
from .stat_tools import zscore_from_pval

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "aggregate_quantiles",
    "clustered_inference",
    "dcrt_zero",
    "desparsified_lasso",
    "desparsified_lasso_pvalue",
    "desparsified_group_lasso",
    "desparsified_group_lasso_pvalue",
    "ensemble_clustered_inference",
    "reid",
    "hd_inference",
    "knockoff_aggregation",
    "model_x_knockoff",
    "multivariate_1D_simulation",
    "permutation_test",
    "permutation_test_pval",
    "reid",
    "empirical_thresholding",
    "zscore_from_pval",
    "CPI",
    "LOCO",
    "PermutationImportance",
]
