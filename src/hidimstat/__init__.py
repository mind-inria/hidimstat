from .base_perturbation import BasePerturbation
from .clustered_inference import clustered_inference, hd_inference
from .desparsified_lasso import (
    desparsified_lasso,
    desparsified_lasso_pvalue,
    desparsified_group_lasso_pvalue,
)
from .conditional_permutation_importance import CPI
from .empirical_thresholding import empirical_thresholding
from .ensemble_clustered_inference import ensemble_clustered_inference
from .knockoffs import (
    model_x_knockoff,
    model_x_knockoff_pvalue,
    model_x_knockoff_bootstrap_quantile,
    model_x_knockoff_bootstrap_e_value,
)
from .leave_one_covariate_out import LOCO
from .noise_std import reid
from .permutation_importance import PermutationImportance
from .statistical_tools.aggregation import quantile_aggregation
from .dcrt import dcrt_zero, dcrt_pvalue
from .permutation_test import permutation_test, permutation_test_pval

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "aggregate_quantiles",
    "quantile_aggregation",
    "clustered_inference",
    "dcrt_zero",
    "dcrt_pvalue",
    "desparsified_lasso",
    "desparsified_lasso_pvalue",
    "desparsified_group_lasso_pvalue",
    "ensemble_clustered_inference",
    "reid",
    "hd_inference",
    "model_x_knockoff",
    "model_x_knockoff_pvalue",
    "model_x_knockoff_bootstrap_quantile",
    "model_x_knockoff_bootstrap_e_value",
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
