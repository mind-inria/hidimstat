import warnings

from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd


class BaseVariableImportance(BaseEstimator):
    """
    Base class for variable importance methods.

    This class provides a foundation for implementing variable importance methods,
    including feature selection based on importance scores and p-values.

    Attributes
    ----------
    importances_ : array-like of shape (n_features,), default=None
        The computed importance scores for each feature.
    pvalues_ : array-like of shape (n_features,), default=None
        The computed p-values for each feature.
    selections_ : array-like of shape (n_features,), default=None
        Binary mask indicating selected features.

    Methods
    -------
    selection(k_best=None, percentile=None, threshold=None, threshold_pvalue=None)
        Selects features based on importance scores and/or p-values using various criteria.

    _check_importance()
        Checks if importance scores and p-values have been computed.
    """

    def __init__(self):
        super().__init__()
        self.importances_ = None
        self.pvalues_ = None
        self.selections_ = None

    def selection(
        self, k_best=None, percentile=None, threshold=None, threshold_pvalue=None
    ):
        """
        Selects features based on variable importance.

        Parameters
        ----------
        k_best : int, optional, default=None
            Selects the top k features based on importance scores.
        percentile : float, optional, default=None
            Selects features based on a specified percentile of importance scores.
        threshold : float, optional, default=None
            Selects features with importance scores above the specified threshold.
        threshold_pvalue : float, optional, default=None
            Selects features with p-values below the specified threshold.

        Returns
        -------
        selection : array-like of shape (n_features,)
            Binary array indicating the selected features.
        """
        self._check_importance()
        if k_best is not None:
            if not isinstance(k_best, str) and k_best > self.importances_.shape[1]:
                warnings.warn(
                    f"k={k_best} is greater than n_features={self.importances_.shape[1]}. "
                    "All the features will be returned."
                )
            assert k_best > 0, "k_best needs to be positive and not null"
        if percentile is not None:
            assert (
                0 < percentile and percentile < 100
            ), "percentile needs to be between 0 and 100"
        if threshold_pvalue is not None:
            assert (
                0 < threshold_pvalue and threshold_pvalue < 1
            ), "threshold_pvalue needs to be between 0 and 1"

        # base on SelectKBest of Scikit-Learn
        if k_best == "all":
            mask_k_best = np.ones(self.importances_.shape, dtype=bool)
        elif k_best == 0:
            mask_k_best = np.zeros(self.importances_.shape, dtype=bool)
        elif k_best is not None:
            mask_k_best = np.zeros(self.importances_.shape, dtype=bool)

            # Request a stable sort. Mergesort takes more memory (~40MB per
            # megafeature on x86-64).
            mask_k_best[np.argsort(self.importances_, kind="mergesort")[-k_best:]] = 1
        else:
            mask_k_best = np.ones(self.importances_.shape, dtype=bool)

        # base on SelectPercentile of Scikit-Learn
        if percentile == 100:
            mask_percentile = np.ones(len(self.importances_), dtype=bool)
        elif percentile == 0:
            mask_percentile = np.zeros(len(self.importances_), dtype=bool)
        elif percentile is not None:
            threshold = np.percentile(self.importances_, 100 - percentile)
            mask_percentile = self.importances_ > threshold
            ties = np.where(self.importances_ == threshold)[0]
            if len(ties):
                max_feats = int(len(self.importances_) * percentile / 100)
                kept_ties = ties[: max_feats - mask_percentile.sum()]
                mask_percentile[kept_ties] = True
        else:
            mask_percentile = np.ones(self.importances_.shape, dtype=bool)

        if threshold is not None:
            mask_threshold = self.importances_ < threshold
        else:
            mask_threshold = np.ones(self.importances_.shape, dtype=bool)

        # base on SelectFpr of Scikit-Learn
        if threshold_pvalue is not None:
            mask_threshold_pvalue = self.pvalues_ < threshold_pvalue
        else:
            mask_threshold_pvalue = np.ones(self.importances_.shape, dtype=bool)

        self.selections_ = (
            mask_k_best & mask_percentile & mask_threshold & mask_threshold_pvalue
        )

        return self.selections_

    def _check_importance(self):
        """
        Checks if the importance scores and p-values have been computed.
        """
        if self.importances_ is None or self.pvalues_ is None:
            raise ValueError(
                "The importances need to be called before calling this method"
            )


class BaseVariableImportanceGroup(BaseVariableImportance):
    """
    Base class for variable importance methods using feature groups.

    This class extends `BaseVariableImportance` to support variable importance
    methods that operate on groups of features, enabling group-wise selection
    and importance evaluation.

    Attributes
    ----------
    n_groups : int, default=None
        The number of feature groups.
    groups : dict, default=None
        A dictionary mapping group names or indices to lists of feature indices or names.
    _groups_ids : array-like of shape (n_groups,), default=None
        Internal representation of group indices for each group.

    Methods
    -------
    fit(X, y=None, groups=None)
        Identifies and stores feature groups based on input or provided grouping.
    _check_fit(X)
        Checks if the class has been fitted and validates group-feature correspondence.
    """

    def __init__(self):
        super().__init__()
        self.n_groups = None
        self.groups = None
        self._groups_ids = None

    def fit(self, X, y=None, groups=None):
        """Base fit method for methods using groups. Identifies the groups.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        y: array-like of shape (n_samples,)
            Not used, only present for consistency with the sklearn API.
        groups: dict, optional
            A dictionary where the keys are the group names and the values are the
            list of column names corresponding to each group. If None, the groups are
            identified based on the columns of X.
        """
        if groups is None:
            self.n_groups = X.shape[1]
            self.groups = {j: [j] for j in range(self.n_groups)}
            self._groups_ids = np.array(list(self.groups.values()), dtype=int)
        elif isinstance(groups, dict):
            self.n_groups = len(groups)
            self.groups = groups
            if isinstance(X, pd.DataFrame):
                self._groups_ids = []
                for group_key in self.groups.keys():
                    self._groups_ids.append(
                        [
                            i
                            for i, col in enumerate(X.columns)
                            if col in self.groups[group_key]
                        ]
                    )
            else:
                self._groups_ids = [
                    np.array(ids, dtype=int) for ids in list(self.groups.values())
                ]
        else:
            raise ValueError("groups needs to be a dictionnary")

    def _check_fit(self, X):
        """
        Check if the perturbation method has been properly fitted.

        This method verifies that the perturbation method has been fitted by checking
        if required attributes are set and if the number of features matches
        the grouped variables.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to validate against the fitted model.

        Raises
        ------
        ValueError
            If the method has not been fitted (i.e., if n_groups, groups,
            or _groups_ids attributes are missing).
        AssertionError
            If the number of features in X does not match the total number
            of features in the grouped variables.
        """
        if self.n_groups is None or self.groups is None or self._groups_ids is None:
            raise ValueError(
                "The class is not fitted. The fit method must be called"
                " to set variable groups. If no grouping is needed,"
                " call fit with groups=None"
            )
        count = 0
        for index_variables in self.groups.values():
            if type(index_variables[0]) is int:
                assert np.all(
                    np.array(index_variables, dtype=int) < X.shape[1]
                ), "X does not correspond to the fitting data."
            count += len(index_variables)
        if X.shape[1] > count:
            warnings.warn(
                "The importance will be computed only for features in the groups."
            )
