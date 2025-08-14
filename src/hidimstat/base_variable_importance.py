import warnings

from sklearn.base import BaseEstimator
import numpy as np

from hidimstat.statistical_tools.multiple_testing import fdr_threshold
from hidimstat.statistical_tools.aggregation import quantile_aggregation


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

    def _check_importance(self):
        """
        Checks if the importance scores and p-values have been computed.
        """
        if self.importances_ is None or self.pvalues_ is None:
            raise ValueError(
                "The importances need to be called before calling this method"
            )


class SelectionBase:
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


class SelectionFDR(SelectionBase):
    def __init__(self):
        self.test_scores_ = None

    def selection_fdr(
        self,
        fdr,
        fdr_control="bhq",
        evalues=False,
        reshaping_function=None,
        adaptive_aggregation=False,
        gamma=0.5,
    ):
        self._check_importance()
        assert self.test_scores_ is not None

        if not evalues:
            assert fdr_control != "ebh", "for p-value, the fdr control can't be 'ebh'"
            pvalues = np.array(
                [
                    _empirical_pval(self.test_scores_[i])
                    for i in range(len(self.test_scores_))
                ]
            )
            aggregated_pval = quantile_aggregation(
                pvalues, gamma=gamma, adaptive=adaptive_aggregation
            )
            threshold = fdr_threshold(
                aggregated_pval,
                fdr=fdr,
                method=fdr_control,
                reshaping_function=reshaping_function,
            )
            selected = aggregated_pval <= threshold
        else:
            assert fdr_control == "ebh", "for e-value, the fdr control need to be 'ebh'"
            ko_threshold = []
            for test_score in self.test_scores_:
                ko_threshold.append(_estimated_threshold(test_score, fdr=fdr))
            evalues = np.array(
                [
                    _empirical_eval(self.test_scores_[i], ko_threshold[i])
                    for i in range(len(self.test_scores_))
                ]
            )
            aggregated_eval = np.mean(evalues, axis=0)
            threshold = fdr_threshold(
                aggregated_eval,
                fdr=fdr,
                method=fdr_control,
                reshaping_function=reshaping_function,
            )
            selected = aggregated_eval >= threshold
        return selected


def _estimated_threshold(test_score, fdr=0.1):
    """
    Calculate the threshold based on the procedure stated in the knockoff article.

    Original code:
    https://github.com/msesia/knockoff-filter/blob/master/R/knockoff/R/knockoff_filter.R

    Parameters
    ----------
    test_score : 1D ndarray, shape (n_features, )
        Vector of test statistic.

    fdr : float
        Desired controlled FDR (false discovery rate) level.

    Returns
    -------
    threshold : float or np.inf
        Threshold level.
    """
    offset = 1  # Offset equals 1 is the knockoff+ procedure.

    threshold_mesh = np.sort(np.abs(test_score[test_score != 0]))
    np.concatenate(
        [[0], threshold_mesh, [np.inf]]
    )  # if there is no solution, the threshold is inf
    # find the right value of t for getting a good fdr
    # Equation 1.8 of barber2015controlling and 3.10 in Candès 2018
    threshold = 0.0
    for threshold in threshold_mesh:
        false_pos = np.sum(test_score <= -threshold)
        selected = np.sum(test_score >= threshold)
        if (offset + false_pos) / np.maximum(selected, 1) <= fdr:
            break
    return threshold


def _empirical_pval(test_score):
    """
    Compute the empirical p-values from the test based on knockoff+.

    Parameters
    ----------
    test_score : 1D ndarray, shape (n_features, )
        Vector of test statistics.

    Returns
    -------
    pvals : 1D ndarray, shape (n_features, )
        Vector of empirical p-values.
    """
    pvals = []
    n_features = test_score.size

    offset = 1  # Offset equals 1 is the knockoff+ procedure.

    test_score_inv = -test_score
    for i in range(n_features):
        if test_score[i] <= 0:
            pvals.append(1)
        else:
            pvals.append(
                (offset + np.sum(test_score_inv >= test_score[i])) / n_features
            )

    return np.array(pvals)


def _empirical_eval(test_score, ko_threshold):
    """
    Compute the empirical e-values from the test based on knockoff.

    Parameters
    ----------
    test_score : 1D ndarray, shape (n_features, )
        Vector of test statistics.

    ko_threshold : float
        Threshold level.

    Returns
    -------
    evals : 1D ndarray, shape (n_features, )
        Vector of empirical e-values.
    """
    evals = []
    n_features = test_score.size

    offset = 1  # Offset equals 1 is the knockoff+ procedure.

    for i in range(n_features):
        if test_score[i] < ko_threshold:
            evals.append(0)
        else:
            evals.append(n_features / (offset + np.sum(test_score <= -ko_threshold)))

    return np.array(evals)
