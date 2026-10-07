import warnings

import numpy as np

from hidimstat._utils.utils import find_stack_level
from hidimstat.statistical_tools.multiple_testing import (
    fdr_threshold,
)


def _selection_generic(
    values,
    k_best=None,
    k_lowest=None,
    percentile=None,
    threshold_max=None,
    threshold_min=None,
):
    """
    Helper function for selecting features based on multiple criteria.

    Parameters
    ----------
    values : array-like of shape (n_features,)
        Values to use for feature selection (e.g., importance scores or p-values)
    k_best : int, default=None
        Selects the top k features based on values.
    k_lowest : int, default=None
        Selects the lowest k features based on values.
    percentile : float, default=None
        Selects features based on a specified percentile of values.
    threshold_max : float, default=None
        Selects features with values below the specified maximum threshold.
    threshold_min : float, default=None
        Selects features with values above the specified minimum threshold.

    Returns
    -------
    selection : array-like of shape (n_features,)
        Boolean array indicating the selected features.
    """
    n_criteria = np.sum(
        [
            criteria is not None
            for criteria in [
                k_best,
                k_lowest,
                percentile,
                threshold_max,
                threshold_min,
            ]
        ]
    )
    if n_criteria <= 1:
        raise ValueError("Only support selection based on one criteria.")
    if k_best is not None:
        if k_best <= 0:
            raise ValueError("k_best needs to be positive.")
        if k_best > values.shape[0]:
            warnings.warn(
                f"k={k_best} is greater than n_features={values.shape[0]}. "
                "All the features will be returned.",
                stacklevel=find_stack_level(),
            )
        mask_k_best = np.zeros_like(values, dtype=bool)

        # based on SelectKBest in Scikit-Learn
        # Request a stable sort. Mergesort takes more memory (~40MB per
        # megafeature on x86-64).
        mask_k_best[np.argsort(values, kind="mergesort")[-k_best:]] = 1
        return mask_k_best
    elif k_lowest is not None:
        if k_lowest <= 0:
            raise ValueError("k_lowest needs to be positive.")
        if k_lowest > values.shape[0]:
            warnings.warn(
                f"k={k_lowest} is greater than n_features={values.shape[0]}. "
                "All the features will be returned.",
                stacklevel=find_stack_level(),
            )
        mask_k_lowest = np.zeros_like(values, dtype=bool)

        # based on SelectKBest in Scikit-Learn
        # Request a stable sort. Mergesort takes more memory (~40MB per
        # megafeature on x86-64).
        mask_k_lowest[np.argsort(values, kind="mergesort")[:k_lowest]] = 1
        return mask_k_lowest
    elif percentile is not None:
        if percentile < 0 or percentile > 100:
            raise ValueError(
                f"percentile must be between 0 and 100 (exclusive). Got {percentile}."
            )
        # based on SelectPercentile in Scikit-Learn
        threshold_percentile = np.percentile(values, 100 - percentile)
        mask_percentile = values > threshold_percentile
        ties = np.where(values == threshold_percentile)[0]
        if len(ties):
            max_feats = int(len(values) * percentile / 100)
            kept_ties = ties[: max_feats - mask_percentile.sum()]
            mask_percentile[kept_ties] = True
        return mask_percentile
    elif threshold_max is not None:
        mask_threshold_max = values < threshold_max
        return mask_threshold_max
    elif threshold_min is not None:
        mask_threshold_min = values > threshold_min
        return mask_threshold_min
    else:
        no_mask = np.ones_like(values, dtype=bool)
        return no_mask


def _selection_fdr(
    p_values,
    importances,
    fdr,
    fdr_control="bhq",
    reshaping_function=None,
    two_tailed_test=False,
):
    """
    Performs feature selection based on False Discovery Rate (FDR) control.

    Parameters
    ----------
    p_values : array-like of shape (n_features,)
        Values to use for feature selection.
    importances : array-like of shape (n_features,) or (n_features, n_folds)
        Importance values to determine selection with positive or negative effect.
    fdr : float
        The target false discovery rate level (between 0 and 1)
    fdr_control: {'bhq', 'bhy'}, default='bhq'
        The FDR control method to use:
        - 'bhq': Benjamini-Hochberg procedure
        - 'bhy': Benjamini-Hochberg-Yekutieli procedure
    reshaping_function: callable or None, default=None
        Optional reshaping function for FDR control methods.
        If None, defaults to sum of reciprocals for 'bhy'.
    two_tailed_test: bool, default=False
        If True, performs two-tailed test selection using both p-values
        for positive effects and one-minus p-values for negative effects. The sign
        of the effect is determined from the sign of the importance scores.

    Returns
    -------
    selected : ndarray of int of shape (n_features,)
        Integer array indicating the selected features.
        1 indicates selected features with positive effects,
        -1 indicates selected features with negative effects,
        0 indicates non-selected features.
    """
    if p_values is None or not isinstance(p_values, np.ndarray):
        raise ValueError(r"'p_values' must be an numpy ndarray.")
    if importances is None or not isinstance(importances, np.ndarray):
        raise ValueError(r"'importances' must be an numpy ndarray.")
    if p_values.shape[0] != importances.shape[0]:
        raise ValueError(
            r"Shape mismatch of 'p_values' and 'importances' on axis 0."
        )
    if fdr < 0 or fdr > 1:
        raise ValueError("'fdr' must be a float between 0 and 1.")
    if fdr_control not in {"bhq", "bhy"}:
        raise ValueError("'fdr_control' must be one of {'bhq', 'bhy'}.")

    # Adjust fdr for two-tailed test
    if two_tailed_test:
        fdr = fdr / 2

    threshold_pvalues = fdr_threshold(
        p_values,
        fdr=fdr,
        method=fdr_control,
        reshaping_function=reshaping_function,
    )
    selected = (p_values <= threshold_pvalues).astype(int)

    # For two-tailed test, determine the sign of the effect
    if two_tailed_test:
        if importances.ndim > 1:
            sign_beta = np.sign(importances.sum(axis=1))
        else:
            sign_beta = np.sign(importances)
        selected = selected * sign_beta
    return selected


def _selection_fwer(
    p_values,
    importances,
    fwer,
    procedure="bonferroni",
    n_tests=1,
    two_tailed_test=False,
):
    """
    Performs feature selection based on False Discovery Rate (FDR) control.

    Parameters
    ----------
    p_values : array-like of shape (n_features,)
        Values to use for feature selection.
    importances : array-like of shape (n_features,) or (n_features, n_folds)
        Importance values to determine selection with positive or negative effect.
    fwer : float
        The target family-wise error rate level (between 0 and 1)
    procedure : {'bonferroni'}, default='bonferroni'
        The FWER control method to use:
        - 'bonferroni': Bonferroni correction
    n_tests : int, default=1
        Factor for multiple testing correction.
    two_tailed_test : bool, default=False
        If True, uses the sign of the importance scores to indicate whether the
        selected features have positive or negative effects.

    Returns
    -------
    selected : ndarray of int of shape (n_features,)
        Integer array indicating the selected features.
        1 indicates selected features with positive effects,
        -1 indicates selected features with negative effects,
        0 indicates non-selected features.

    Raises
    ------
    ValueError
        If `procedure` is not 'bonferroni'.
        If 'n_tests' <= 0.
    """
    if p_values is None or not isinstance(p_values, np.ndarray):
        raise ValueError(r"'p_values' must be an numpy ndarray.")
    if importances is None or not isinstance(importances, np.ndarray):
        raise ValueError(r"'importances' must be an numpy ndarray.")
    if p_values.shape[0] != importances.shape[0]:
        raise ValueError(
            r"Shape mismatch of 'p_values' and 'importances' on axis 0."
        )
    if fwer < 0 or fwer > 1:
        raise ValueError("'fwer' must be a float between 0 and 1.")
    if procedure != "bonferroni":
        raise ValueError("Only 'bonferroni' procedure is supported")
    if n_tests <= 0:
        raise ValueError(r"'n_tests' cannot be less than or equal to 0.")

    # Adjust fwer for two-tailed test
    if two_tailed_test:
        fwer = fwer / 2

    threshold_pvalue = fwer / n_tests
    selected = (p_values < threshold_pvalue).astype(int)
    if two_tailed_test:
        if importances.ndim > 1:
            sign_beta = np.sign(importances.sum(axis=1))
        else:
            sign_beta = np.sign(importances)
        selected = selected * sign_beta

    return selected
