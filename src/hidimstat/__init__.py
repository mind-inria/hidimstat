from .base_perturbation import BasePerturbation
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
from .knockoffs import (
    model_x_knockoff,
    model_x_knockoff_pvalue,
    model_x_knockoff_bootstrap_quantile,
    model_x_knockoff_bootstrap_e_value,
)
from .leave_one_covariate_out import LOCO
from .noise_std import reid
from .permutation_importance import PermutationImportance
from .dcrt import dcrt_zero, dcrt_pvalue
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
    "dcrt_pvalue",
    "desparsified_lasso",
    "desparsified_lasso_pvalue",
    "desparsified_group_lasso_pvalue",
    "reid",
<<<<<<< HEAD
    "knockoff_aggregation",
||||||| fa4a7f5
    "hd_inference",
    "knockoff_aggregation",
=======
    "hd_inference",
>>>>>>> main
    "model_x_knockoff",
<<<<<<< HEAD
||||||| fa4a7f5
    "multivariate_1D_simulation",
=======
    "model_x_knockoff_pvalue",
    "model_x_knockoff_bootstrap_quantile",
    "model_x_knockoff_bootstrap_e_value",
    "multivariate_1D_simulation",
>>>>>>> main
    "permutation_test",
    "permutation_test_pval",
    "empirical_thresholding",
    "CPI",
    "LOCO",
    "PermutationImportance",
]
