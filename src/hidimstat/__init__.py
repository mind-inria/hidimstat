from .ensemble_clustered_inference import (
    clustered_inference,
    clustered_inference_pvalue,
)
from .ensemble_clustered_inference import (
    ensemble_clustered_inference,
    ensemble_clustered_inference_pvalue,
)
from .desparsified_lasso import (
    desparsified_lasso,
    desparsified_lasso_pvalue,
    desparsified_group_lasso_pvalue,
)
from .conditional_permutation_importance import CPI
from .empirical_thresholding import empirical_thresholding
from .knockoff_aggregation import knockoff_aggregation
from .knockoffs import model_x_knockoff
from .leave_one_covariate_out import LOCO
from .noise_std import reid
from .permutation_importance import PermutationImportance
from .permutation_test import permutation_test, permutation_test_pval

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "clustered_inference",
    "clustered_inference_pvalue",
    "ensemble_clustered_inference",
    "ensemble_clustered_inference_pvalue",
    "dcrt_zero",
    "desparsified_lasso",
    "desparsified_lasso_pvalue",
    "desparsified_group_lasso",
    "desparsified_group_lasso_pvalue",
    "reid",
    "knockoff_aggregation",
    "model_x_knockoff",
    "permutation_test",
    "permutation_test_pval",
    "empirical_thresholding",
    "CPI",
    "LOCO",
    "PermutationImportance",
]
