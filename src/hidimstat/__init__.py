from .adaptive_permutation_threshold import ada_svr
from .clustered_inference import clustered_inference, hd_inference
from .desparsified_lasso import desparsified_group_lasso, desparsified_lasso
from .Dnn_learner_single import DnnLearnerSingle
from .ensemble_clustered_inference import ensemble_clustered_inference
from .knockoffs import (
    model_x_knockoff,
    model_x_knockoff_aggregation,
    model_x_knockoff_filter,
    model_x_knockoff_pvalue,
    model_x_knockoff_bootstrap_quantile,
    model_x_knockoff_bootstrap_e_value,
)
from .multi_sample_split import aggregate_quantiles
from .noise_std import group_reid, reid
from .permutation_test import permutation_test_cv
from .scenario import multivariate_1D_simulation
from .standardized_svr import standardized_svr
from .stat_tools import zscore_from_pval
from .cpi import CPI
from .loco import LOCO
from .permutation_importance import PermutationImportance

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "ada_svr",
    "aggregate_quantiles",
    "clustered_inference",
    "dcrt_zero",
    "desparsified_lasso",
    "desparsified_group_lasso",
    "DnnLearnerSingle",
    "ensemble_clustered_inference",
    "group_reid",
    "hd_inference",
    "model_x_knockoff",
    "model_x_knockoff_aggregation",
    "model_x_knockoff_filter",
    "model_x_knockoff_pvalue",
    "model_x_knockoff_bootstrap_quantile",
    "model_x_knockoff_bootstrap_e_value",
    "multivariate_1D_simulation",
    "permutation_test_cv",
    "reid",
    "standardized_svr",
    "zscore_from_pval",
    "CPI",
    "LOCO",
    "PermutationImportance",
]
