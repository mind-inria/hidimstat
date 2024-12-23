from .ada_svr import ada_svr, ada_svr_pvalue
from .clustered_inference import clustered_inference, hd_inference
from .desparsified_lasso import desparsified_group_lasso, desparsified_lasso
from .Dnn_learner_single import DnnLearnerSingle
from .ensemble_clustered_inference import ensemble_clustered_inference
from .knockoff_aggregation import knockoff_aggregation
from .knockoffs import model_x_knockoff
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
    "ada_svr_pvalue",
    "aggregate_quantiles",
    "clustered_inference",
    "dcrt_zero",
    "desparsified_lasso",
    "desparsified_group_lasso",
    "DnnLearnerSingle",
    "ensemble_clustered_inference",
    "group_reid",
    "hd_inference",
    "knockoff_aggregation",
    "model_x_knockoff",
    "multivariate_1D_simulation",
    "permutation_test_cv",
    "reid",
    "standardized_svr",
    "zscore_from_pval",
    "CPI",
    "LOCO",
    "PermutationImportance",
]
