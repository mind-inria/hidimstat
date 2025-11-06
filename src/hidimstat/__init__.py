from .conditional_feature_importance import CFI, cfi_analysis
from .desparsified_lasso import (
    desparsified_group_lasso_pvalue,
    desparsified_lasso,
    desparsified_lasso_pvalue,
)
from .distilled_conditional_randomization_test import D0CRT, d0crt
from .ensemble_clustered_inference import (
    clustered_inference,
    clustered_inference_pvalue,
    ensemble_clustered_inference,
    ensemble_clustered_inference_pvalue,
)
from .knockoffs import ModelXKnockoff
from .leave_one_covariate_out import LOCO, loco_analysis
from .noise_std import reid
from .permutation_feature_importance import PFI, pfi_analysis
from .statistical_tools.aggregation import quantile_aggregation

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "quantile_aggregation",
    "clustered_inference",
    "clustered_inference_pvalue",
    "ensemble_clustered_inference",
    "ensemble_clustered_inference_pvalue",
    "d0crt",
    "D0CRT",
    "desparsified_lasso",
    "desparsified_lasso_pvalue",
    "desparsified_group_lasso_pvalue",
    "reid",
    "ModelXKnockoff",
    "CFI",
    "cfi_analysis",
    "LOCO",
    "loco_analysis",
    "PFI",
    "pfi_analysis",
]
