"""
Sage values
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from hidimstat._utils.utils import check_random_state
from hidimstat.base_variable_importance import (
    BaseVariableImportance,
    GroupVariableImportanceMixin,
)


def _sage_value_function(
    estimator,
    X,
    y,
    subset,
    method,
    loss,
    n_permutations,
    imputation="marginal",
    random_state=None,
):
    """
    Compute the SAGE value function for a given subset of features. The
    complement of the subset is sampled according to the specified imputation
    strategy. Then the loss is computed for the perturbed data.
    """
    rng = check_random_state(random_state)

    if imputation != "marginal":
        raise NotImplementedError(
            "Only marginal imputation is currently implemented for SAGE values."
        )

    # Sample the complement of the subset according to the marginal distribution
    n_samples, n_features = X.shape
    complement = np.setdiff1d(np.arange(n_features), subset)
    X_sampled = np.tile(
        X, (n_permutations, 1, 1)
    )  # (n_perm, n_samples, n_feat)
    for perm_idx in range(n_permutations):
        for col in complement:
            X_sampled[perm_idx, :, col] = rng.permutation(X[:, col])
    X_sampled_batch = X_sampled.reshape(-1, n_features)
    y_pred = getattr(estimator, method)(X_sampled_batch)
    # In case of classification, the output is a 2D array. Reshape accordingly
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(n_permutations, n_samples)
    else:
        y_pred = y_pred.reshape(n_permutations, n_samples, y_pred.shape[1])

    losses = np.array([loss(y, y_pred[i]) for i in range(n_permutations)])
    return subset, np.mean(losses)


def _sample_feature_subsets(n_features, j, n_subsets, random_state=None):
    r"""
    Helper function to sample random subsets S in {0..p-1} \ {feature_idx} for
    Shapley estimation. Each subset size k is sampled uniformly. Then, for a
    given size k, there are comb(p-1, k) subsets of size k from which we sample
    uniformly. This Monte Carlo sampling procedure accounts for the reweighting
    terms in the Shapley formula.
    """
    minus_j = np.delete(np.arange(n_features), j)

    rng = check_random_state(random_state)
    subsets = [
        np.sort(rng.choice(minus_j, size=k, replace=False))
        for k in rng.integers(0, len(minus_j) + 1, size=n_subsets)
    ]
    return subsets


class SAGE(BaseVariableImportance, GroupVariableImportanceMixin):
    """
    Shaple Additive Global Importance (SAGE) values for feature importance.

    Parameters
    ----------
    estimator : object
        The fitted model for which to compute the SAGE values.
    method : str, default="predict"
        The method of the estimator to use for predictions.
    loss : callable, default=mean_squared_error
        The loss function to use for computing the SAGE values.
    imputation : str, default="marginal"
        The imputation strategy to use for sampling the complement of the
        subset.
    n_subsets : int, default=100
        The number of subsets to sample for estimating the SAGE values, for
        each feature.
    n_permutations : int, default=50
        The number of samples to draw using the `imputation` strategy for each
        subset (the samples are drawn from the complement of the subset).
    features_groups : dict, optional
        Group of features for which to compute the importance. Default is None,
        which means that the importance of each individual feature will be
        computed.
    random_state : int or RandomState, optional
        Random state for reproducibility. Default is None.
    n_jobs : int, default=1
        The number of parallel jobs to run for computing the SAGE values.
    """

    def __init__(
        self,
        estimator,
        method="predict",
        loss=mean_squared_error,
        imputation="marginal",
        n_subsets=50,
        n_permutations=50,
        features_groups=None,
        random_state=None,
        n_jobs=1,
    ):
        super().__init__()
        GroupVariableImportanceMixin.__init__(
            self, features_groups=features_groups
        )
        self.estimator = estimator
        self.method = method
        self.loss = loss
        self.imputation = imputation
        self.n_subsets = n_subsets
        self.n_permutations = n_permutations
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.estimator_ = self._initial_fit(self.estimator, X, y)
        GroupVariableImportanceMixin.fit(self, X, y)
        self._value_cache = {}
        return self

    def importance(self, X, y):
        self._check_fit()
        rng = check_random_state(self.random_state)

        # 1. Sample all subsets, create a dictionary mapping between subsets
        # and corresponding SAGE value function to avoid redundant computation.

        # For each feature, list of tuples (S, S U {j})
        self.sum_terms_ = {j: [] for j in range(self.n_features_groups_)}
        # dictionary to store computed SAGE values for subsets
        self.subset_value_map_ = {}
        # subset of features without j
        self.subset_map_ = {}
        for j in range(self.n_features_groups_):
            group_ids = self.features_groups_[j]
            subsets = _sample_feature_subsets(
                X.shape[1], group_ids, self.n_subsets, random_state=rng
            )
            for S in subsets:
                S_key = tuple(S)
                self.subset_value_map_[S_key] = None
                S_j = np.sort(np.union1d(S, group_ids))
                S_j_key = tuple(S_j)
                self.subset_value_map_[S_j_key] = None
                self.sum_terms_[j].append((S_key, S_j_key))
                self.subset_map_[S_key] = S
                self.subset_map_[S_j_key] = S_j

        # 2. Compute SAGE values for all unique subsets
        sage_val_res = Parallel(n_jobs=self.n_jobs)(
            delayed(_sage_value_function)(
                self.estimator_,
                X,
                y,
                subset=self.subset_map_[subset_key],
                method=self.method,
                loss=self.loss,
                n_permutations=self.n_permutations,
                imputation=self.imputation,
                random_state=rng,
            )
            for subset_key in tqdm(self.subset_value_map_)
        )

        # 3. Collect results and compute the sum
        for subset, value in sage_val_res:
            subset_key = tuple(subset)
            self.subset_value_map_[subset_key] = value

        self.importances_ = np.zeros(self.n_features_groups_)
        for j in range(self.n_features_groups_):
            for S_key, S_j_key in self.sum_terms_[j]:
                self.importances_[j] += (
                    self.subset_value_map_[S_key]
                    - self.subset_value_map_[S_j_key]
                )
            self.importances_[j] /= len(self.sum_terms_[j])

        return self.importances_
