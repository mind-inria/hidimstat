from .ensemble_clustered_inference import clustered_inference, clustered_inference_pvalue
from .ensemble_clustered_inference import ensemble_clustered_inference, ensemble_clustered_inference_pvalue
from .desparsified_lasso import (
    desparsified_lasso,
    desparsified_lasso_pvalue,
    desparsified_group_lasso_pvalue,
)
from .Dnn_learner_single import DnnLearnerSingle
from .knockoff_aggregation import knockoff_aggregation
from .knockoffs import model_x_knockoff
from .noise_std import reid
from .permutation_test import permutation_test, permutation_test_pval
from .scenario import multivariate_1D_simulation
from .stat_tools import zscore_from_pval, aggregate_quantiles, aggregate_medians
from .empirical_thresholding import empirical_thresholding
from .cpi import CPI
from .loco import LOCO
from .permutation_importance import PermutationImportance

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "aggregate_quantiles",
    "aggregate_medians",
    "clustered_inference",
    "clustered_inference_pvalue",
    "ensemble_clustered_inference",
    "ensemble_clustered_inference_pvalue",
    "dcrt_zero",
    "desparsified_lasso",
    "desparsified_lasso_pvalue",
    "desparsified_group_lasso_pvalue",
    "DnnLearnerSingle",
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
