from collections.abc import Iterable
from tqdm import tqdm
from typing import override
import warnings

from joblib import Parallel, delayed
import numpy as np
from scipy import sparse
from scipy.stats.mstats import mquantiles
from sklearn.base import is_classifier, is_regressor
from sklearn.utils import _safe_indexing, check_array, check_random_state
from sklearn.utils._indexing import (
    _determine_key_type,
    _get_column_indices,
    _safe_assign,
)
from sklearn.utils._response import _get_response_values
from sklearn.utils.validation import _check_sample_weight, check_is_fitted
from sklearn.inspection._pd_utils import _check_feature_names, _get_feature_index
from sklearn.inspection._partial_dependence import _grid_from_X

from hidimstat.base_variable_importance import (
    BaseVariableImportance,
)


def _grid_from_X(
    X,
    percentiles,
    is_categorical,
    grid_resolution,
    custom_values,
    resolution_statistique=False,
):
    """
    Generate a grid of points based on the percentiles of X.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_target_features)
        The data from which to generate the grid.

    percentiles : tuple of float (p1, p2)
        The lower and upper percentiles to use for the grid boundaries.
        Must satisfy 0 <= p1 < p2 <= 1.

    is_categorical : list of bool
        For each feature, indicates whether it is categorical or not.
        For categorical features, unique values are used instead of percentiles.

    grid_resolution : int
        Number of equally spaced points for the grid.
        Must be greater than 1.

    custom_values : dict
        Mapping from column index of X to array-like of custom values
        to use for that feature instead of the generated grid.

    resolution_statistique : bool, default=False
        If True, uses quantiles for grid points instead of equally spaced points
        between percentile boundaries.

    Returns
    -------
    values : list of 1d ndarrays
        List containing the unique grid values for each feature.
        Each array has length <= grid_resolution.

    indexes : list of 1d ndarrays
        For each feature, contains the indices mapping each sample in X
        to its position in the grid.

    Notes
    -----
    Based on scikit-learn's _grid_from_X implementation:
    https://github.com/scikit-learn/scikit-learn/blob/c5497b7f7eacfaff061cf68e09bcd48aa93d4d6b/sklearn/inspection/_partial_dependence.py#L40
    """
    if not isinstance(percentiles, Iterable) or len(percentiles) != 2:
        raise ValueError("'percentiles' must be a sequence of 2 elements.")
    if not all(0 <= x <= 1 for x in percentiles):
        raise ValueError("'percentiles' values must be in [0, 1].")
    if percentiles[0] >= percentiles[1]:
        raise ValueError("percentiles[0] must be strictly less than percentiles[1].")

    if grid_resolution <= 1:
        raise ValueError("'grid_resolution' must be strictly greater than 1.")

    def _convert_custom_values(values):
        # Convert custom types such that object types are always used for string arrays
        dtype = object if any(isinstance(v, str) for v in values) else None
        return np.asarray(values, dtype=dtype)

    custom_values = {k: _convert_custom_values(v) for k, v in custom_values.items()}
    if any(v.ndim != 1 for v in custom_values.values()):
        error_string = ", ".join(
            f"Feature {k}: {v.ndim} dimensions"
            for k, v in custom_values.items()
            if v.ndim != 1
        )

        raise ValueError(
            "The custom grid for some features is not a one-dimensional array. "
            f"{error_string}"
        )

    values = []
    indexes = []
    # TODO: we should handle missing values (i.e. `np.nan`) specifically and store them
    # in a different Bunch attribute.
    for feature_idx, is_cat in enumerate(is_categorical):
        data = _safe_indexing(X, feature_idx, axis=1)
        if feature_idx in custom_values:
            # Use values in the custom range
            axis = custom_values[feature_idx]
        else:
            try:
                uniques = np.unique(data)
            except TypeError as exc:
                # `np.unique` will fail in the presence of `np.nan` and `str` categories
                # due to sorting. Temporary, we reraise an error explaining the problem.
                raise ValueError(
                    f"The column #{feature_idx} contains mixed data types. Finding unique "
                    "categories fail due to sorting. It usually means that the column "
                    "contains `np.nan` values together with `str` categories. Such use "
                    "case is not yet supported in scikit-learn."
                ) from exc

            if is_cat or uniques.shape[0] < grid_resolution:
                # Use the unique values either because:
                # - feature has low resolution use unique values
                # - feature is categorical
                axis = uniques
            else:
                # create axis based on percentiles and grid resolution
                if resolution_statistique:
                    axis = np.unique(
                        mquantiles(
                            data,
                            prob=np.linspace(0.0, 1.0, grid_resolution + 1),
                            axis=0,
                        )
                    )
                else:
                    emp_percentiles = mquantiles(data, prob=percentiles, axis=0)
                    if np.allclose(emp_percentiles[0], emp_percentiles[1]):
                        raise ValueError(
                            "percentiles are too close to each other, "
                            "unable to build the grid. Please choose percentiles "
                            "that are further apart."
                        )
                    axis = np.linspace(
                        emp_percentiles[0],
                        emp_percentiles[1],
                        num=grid_resolution,
                        endpoint=True,
                    )
        values.append(axis)
        if not is_cat:
            # -1 is for correction of the number of classes
            digitize = np.digitize(data, axis, right=True) - 1
        else:
            # convert the object in string if it's necesarry
            string_to_num = {s: i for i, s in enumerate(sorted(set(data)))}
            digitize = np.digitize(
                np.array([string_to_num[s] for s in data]),
                [string_to_num[s] for s in axis],
                right=True,
            )
        indexes.append(np.clip(digitize, 0, None))
    return values, indexes


def _partial_dependence_brute(est, grid, features, X, method, n_jobs):
    """
    Calculate partial dependence using the brute force method.

    For each value in `grid`, replaces the target features with that value for all samples
    in X, makes predictions, and averages them. This computes the mean model response
    across the data distribution for each grid point.

    Parameters
    ----------
    est : BaseEstimator
        A fitted estimator implementing predict, predict_proba, or decision_function.
        Multioutput-multiclass classifiers not supported.

    grid : list of 1d arrays
        List containing grid values for each target feature. Each array contains the
        values where partial dependence will be evaluated.

    features : array-like of int
        Indices of the features for which to compute partial dependence.

    X : array-like of shape (n_samples, n_features)
        Training data used to compute predictions.

    method : {'auto', 'predict_proba', 'decision_function'}, default='auto'
        Method to get model predictions:
        - 'auto': uses predict_proba if available, otherwise decision_function
        - 'predict_proba': uses predict_proba
        - 'decision_function': uses decision_function
        For regressors, always uses predict.

    n_jobs : int
        Number of parallel jobs to run.

    Returns
    -------
    list of arrays
        List containing arrays of predictions for each feature, averaged over X.
        For each feature:
        - Shape (n_grid_points, n_samples) for regression/binary classification
        - Shape (n_grid_points, n_samples, n_classes) for multiclass
    """
    predictions = []

    if method == "auto":
        method = (
            "predict" if is_regressor(est) else ["predict_proba", "decision_function"]
        )
    predictions = Parallel(n_jobs=n_jobs)(
        delayed(_joblib_get_predictions)(variable, values, X, est, method)
        for variable, values in tqdm(zip(features, grid))
    )
    return predictions


def _joblib_get_predictions(variable, values, X, est, method):
    """
    Helper function to get predictions for a single feature for parallel processing.

    Parameters
    ----------
    variable : int
        Index of the feature for which to get predictions.
    values : array-like
        Grid values for the target feature.
    X : array-like of shape (n_samples, n_features)
        Training data used to compute predictions.
    est : BaseEstimator
        A fitted estimator implementing predict, predict_proba, or decision_function.
    method : str or list of str
        Method to get model predictions: 'predict', 'predict_proba', or 'decision_function'.

    Returns
    -------
    list of array
        List of predictions for each grid point, each with shape:
        - (n_samples,) for regression
        - (n_samples, 1) for binary classification
        - (n_samples, n_classes) for multiclass
    """
    predictions = []
    for new_values in values:
        _safe_assign(X, new_values, column_indexer=variable)

        # Note: predictions is of shape
        # (n_points,) for non-multioutput regressors
        # (n_points, n_tasks) for multioutput regressors
        # (n_points, 1) for the regressors in cross_decomposition (I think)
        # (n_points, 1) for binary classification (positive class already selected)
        # (n_points, n_classes) for multiclass classification
        pred, _ = _get_response_values(est, X, response_method=method)
        if len(pred.shape) > 1 and pred.shape[1] > 1:
            pred = np.max(pred, axis=1)
        predictions.append(np.squeeze(pred))
    return predictions


class PartialDependancePlot(BaseVariableImportance):
    """
    Partial Dependence Plot (PDP):footcite:t:`friedman2001greedy` for analyzing
    feature effects on model predictions. This is based on individual conditional
    expectation (ICE) :footcite:t:`goldstein2015peeking`.

    PDP shows the average model prediction across different values of target features,
    while marginalizing over the values of all other features. It helps understand
    how features affect predictions on average. ICE curves show predictions for
    individual samples as the feature value changes.

    Feature importance scores are computed following :footcite:t:`greenwell2018simple`:
    - For continuous features: standard deviation of PDP curve
    - For categorical features: range of PDP values divided by 4

    Parameters
    ----------
    estimator : BaseEstimator
        A fitted estimator object implementing predict, predict_proba, or
        decision_function. Must be a regressor or binary/multiclass classifier.
        Multioutput-multiclass classifiers are not supported.

    features : array-like of {int, str}, default=None
        Features for which to compute partial dependence. Can be:
        - Single feature: int or str
        - Multiple features: list of int or str
        Feature indices or names must match training data.

    method : {'auto', 'predict_proba', 'decision_function'}, default='auto'
        Method for getting predictions:
        - 'auto': tries predict_proba first, falls back to decision_function
        - 'predict_proba': uses predicted probabilities
        - 'decision_function': uses decision function scores
        Ignored for regressors which always use predict.

    n_jobs : int, default=1
        Number of CPU cores to use for parallel processing.
        -1 means using all processors.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights for computing weighted averages of predictions.
        If None, samples are weighted equally.

    categorical_features : array-like, default=None
        Specifies categorical features. Can be:
        - None: no categorical features
        - boolean array: mask indicating categorical features
        - array of int/str: indices/names of categorical features

    feature_names : array-like of str, default=None
        Names of features in the training data.
        If None, uses array indices for NumPy arrays or column names for pandas.

    percentiles : tuple of float, default=(0.05, 0.95)
        Lower and upper percentiles for grid boundaries.
        Used for continuous features if custom_values not provided.
        Must be in [0, 1].

    grid_resolution : int, default=100
        Number of grid points for continuous features.
        Higher values give more granular curves but increase computation time.
        Ignored if custom_values provided.

    custom_values : dict, default=None
        Custom grid values for features. Dictionary mapping feature index/name
        to array of values to evaluate.
        Overrides percentiles and grid_resolution for specified features.

    resolution_statistique : bool, default=False
        If True, uses quantile-based grid points instead of evenly spaced points.
        Can better capture feature distribution.

    Attributes
    ----------
    importances_ : ndarray of shape (n_features,)
        Computed feature importance scores based on PDP variance

    ices_ : list of arrays
        Individual Conditional Expectation curves for each feature

    values_ : list of arrays
        Grid values used for each feature

    See Also
    --------
    sklearn.inspection.partial_dependence : Similar functionality in scikit-learn

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        estimator,
        features=None,
        method: str = "auto",
        n_jobs: int = 1,
        sample_weight=None,
        categorical_features=None,
        feature_names=None,
        percentiles=(0.05, 0.95),
        grid_resolution=100,
        custom_values=None,
        resolution_statistique=False,
    ):
        super().__init__()
        check_is_fitted(estimator)
        self.estimator = estimator
        if not (is_classifier(self.estimator) or is_regressor(self.estimator)):
            raise ValueError("'estimator' must be a fitted regressor or classifier.")

        if is_classifier(self.estimator) and isinstance(
            self.estimator.classes_[0], np.ndarray
        ):
            raise ValueError("Multiclass-multioutput estimators are not supported")

        if is_regressor(self.estimator) and (method != "auto" and method != "predict"):
            raise ValueError(
                "The method parameter is ignored for regressors and "
                "must be 'auto' or 'predict'."
            )
        self.features = features
        self.method = method
        self.n_jobs = n_jobs
        self.sample_weight = sample_weight
        self.categorical_features = categorical_features
        self.feature_names = feature_names
        self.percentiles = percentiles
        self.grid_resolution = grid_resolution
        self.custom_values = custom_values
        self.resolution_statistique = resolution_statistique

    @override
    def fit(self, X=None, y=None):
        """
        Fits the PartialDependencePlot model. This method has no effect as PDP
        only needs a fitted estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            (Not used) Training data. Not used, present here for API consistency.

        y : array-like of shape (n_samples,)
            (Not used) Target values. Not used, present here for API consistency.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if X is not None:
            warnings.warn("X won't be used")
        if y is not None:
            warnings.warn("y won't be used")
        return self

    def _set_enviroment_importance(self, X_):
        """
        Set up environment variables needed for importance calculation.

        Parameters
        ----------
        X_ : array-like of shape (n_samples, n_features)
            Input data on which to compute feature importance.

        Returns
        -------
        tuple
            Contains:
            - X_subset : array-like, subset of X_ with only target features
            - custom_values_for_X_subset : dict, custom values for target features

        Raises
        ------
        ValueError
            If features indices are negative.
        ValueError
            If categorical_features is empty list.
        ValueError
            If categorical_features boolean mask has wrong length.
        ValueError
            If categorical_features has invalid dtype.
        ValueError
            If features contain integer data.
        """
        if self.sample_weight is not None:
            self.sample_weight = _check_sample_weight(self.sample_weight, X_)

        if _determine_key_type(self.features, accept_slice=False) == "int":
            # _get_column_indices() supports negative indexing. Here, we limit
            # the indexing to be positive. The upper bound will be checked
            # by _get_column_indices()
            if np.any(np.less(self.features, 0)):
                raise ValueError(
                    "all features must be in [0, {}]".format(X_.shape[1] - 1)
                )

        if self.features is None:
            self.features = list(range(X_.shape[1]))
        self.features_indices = np.asarray(
            _get_column_indices(X_, self.features), dtype=np.intp, order="C"
        ).ravel()

        self.feature_names = _check_feature_names(X_, self.feature_names)

        n_features = X_.shape[1]
        if self.categorical_features is None:
            self.is_categorical = [False] * len(self.features_indices)
        else:
            categorical_features = np.asarray(self.categorical_features)
            if categorical_features.size == 0:
                raise ValueError(
                    "Passing an empty list (`[]`) to `categorical_features` is not "
                    "supported. Use `None` instead to indicate that there are no "
                    "categorical features."
                )
            if categorical_features.dtype.kind == "b":
                # categorical features provided as a list of boolean
                if categorical_features.size != n_features:
                    raise ValueError(
                        "When `categorical_features` is a boolean array-like, "
                        "the array should be of shape (n_features,). Got "
                        f"{categorical_features.size} elements while `X` contains "
                        f"{n_features} features."
                    )
                self.is_categorical = [
                    categorical_features[idx] for idx in self.features_indices
                ]
            elif categorical_features.dtype.kind in ("i", "O", "U"):
                # categorical features provided as a list of indices or feature names
                categorical_features_idx = [
                    _get_feature_index(cat, feature_names=self.feature_names)
                    for cat in categorical_features
                ]
                self.is_categorical = [
                    idx in categorical_features_idx for idx in self.features_indices
                ]
            else:
                raise ValueError(
                    "Expected `categorical_features` to be an array-like of boolean,"
                    f" integer, or string. Got {categorical_features.dtype} instead."
                )

        custom_values = self.custom_values or {}
        if isinstance(self.features, (str, int)):
            self.features = [self.features]

        for feature_idx, feature, is_cat in zip(
            self.features_indices, self.features, self.is_categorical
        ):
            if is_cat:
                continue

            if _safe_indexing(X_, feature_idx, axis=1).dtype.kind in "iu":
                # TODO(1.9): raise a ValueError instead.
                raise ValueError(
                    f"The column {feature!r} contains integer data. Partial "
                    "dependence plots are not supported for integer data: this "
                    "can lead to implicit roun[val.shape[0] for val in self.values_]ding with NumPy arrays or even errors "
                    "with newer pandas versions. Please convert numerical features"
                    "to floating point dtypes ahead of time to avoid problems. "
                )

        X_subset = _safe_indexing(X_, self.features_indices, axis=1)

        custom_values_for_X_subset = {
            index: custom_values.get(feature)
            for index, feature in enumerate(self.features)
            if feature in custom_values
        }
        return X_subset, custom_values_for_X_subset

    def importance(self, X, y=None):
        """
        Calculate partial dependence importance scores for each feature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to compute partial dependence on. Must have same features
            as training data.

        y : array-like of shape (n_samples,)
            (not used) Target values. Kept for API compatibility.

        Returns
        -------
        ndarray of shape (n_features,)
            Importance scores for each feature based on partial dependence.
            Higher values indicate greater importance.

        Raises
        ------
        ValueError
            If features indices are negative.
        ValueError
            If categorical_features is empty list.
        ValueError
            If categorical_features boolean mask has wrong length.
        ValueError
            If categorical_features has invalid dtype.
        ValueError
            If features contain integer data.
        """
        if y is not None:
            warnings.warn("y won't be used")
        # Use check_array only on lists and other non-array-likes / sparse. Do not
        # convert DataFrame into a NumPy array.
        if not (hasattr(X, "__array__") or sparse.issparse(X)):
            X_ = check_array(X, ensure_all_finite="allow-nan", dtype=object)
        else:
            X_ = X
        X_subset, custom_values_for_X_subset = self._set_enviroment_importance(X_)

        self.values_, _ = _grid_from_X(
            X_subset,
            self.percentiles,
            self.is_categorical,
            self.grid_resolution,
            custom_values_for_X_subset,
            self.resolution_statistique,
        )

        self.ices_ = _partial_dependence_brute(
            self.estimator,
            self.values_,
            self.features_indices,
            X_,
            self.method,
            self.n_jobs,
        )

        averaged_predictions_continious = []
        averaged_predictions_categorie = []
        # average over samples
        for index, ice in enumerate(self.ices_):
            if not self.is_categorical[index]:
                averaged_predictions_continious.append(
                    np.average(ice, axis=1, weights=self.sample_weight)
                )
            else:
                averaged_predictions_categorie.append(
                    np.average(ice, axis=1, weights=self.sample_weight)
                )

        # compute importance from equation 4 of greenwell2018simple
        self.importances_ = np.zeros_like(self.is_categorical, dtype=float)
        # importance for continous variable
        if len(averaged_predictions_continious) > 0:
            importance_continious = []
            for averaged_prediction in averaged_predictions_continious:
                importance_continious.append(
                    np.sqrt(
                        np.sum(
                            (averaged_prediction - np.mean(averaged_prediction)) ** 2
                        )
                        / (len(averaged_prediction) - 1)
                    )
                )
            self.importances_[np.logical_not(self.is_categorical)] = (
                importance_continious
            )
        # importance for categoritcal features
        if len(averaged_predictions_categorie) > 0:
            importance_categories = []
            for averaged_prediction in averaged_predictions_categorie:
                importance_categories.append(
                    (np.max(averaged_prediction) - np.min(averaged_prediction)) / 4
                )
            self.importances_[self.is_categorical] = importance_categories
        self.pvalues_ = None

        return self.importances_

    def fit_importance(self, X, y=None, cv=None):
        """
        Convenience method to fit and calculate importance scores in one step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to compute partial dependence on. Must have same features
            as training data.

        y : array-like of shape (n_samples,), default=None
            Not used, kept for API compatibility.

        cv : object, default=None
            Not used, kept for API compatibility.

        Returns
        -------
        ndarray of shape (n_features,)
            Importance scores for each feature based on partial dependence.
            Higher values indicate greater importance.
        """
        if y is not None:
            warnings.warn("y won't be used")
        if cv is not None:
            warnings.warn("cv won't be used")
        self.fit()
        return self.importance(X)

    def plot(
        self,
        feature_id,
        ax=None,
        X=None,
        nbins=5,
        percentage_ice=1.0,
        random_state=None,
        **kwargs,
    ):
        """
        Plot partial dependence and ICE curves for a given feature.

        Parameters
        ----------
        feature_id : int
            Index of the feature to plot from the features used to compute PDP.

        ax : matplotlib.axes.Axes, default=None
            Pre-existing axes for the plot. If None, generates new figure and axes.

        X : array-like, default=None
            Training data used to compute the quantiles and feature distribution.
            If provided, adds rugplot and quantile indicators to visualization.

        nbins : int, default=5
            Number of bins/quantiles to show in the plot when X is provided.

        percentage_ice : float, default=1.0
            Proportion of ICE curves to plot, between 0 and 1.
            Lower values reduce visual clutter.

        random_state : int or RandomState, default=None
            Controls random sampling of ICE curves when percentage_ice < 1.

        **kwargs : dict
            Additional keyword arguments passed to plot function.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.

        Raises
        ------
        Exception
            If seaborn is not installed.

        Notes
        -----
        For continuous features:
        - Blue lines show individual ICE curves
        - Black line shows averaged PDP curve
        - Rug plot shows data distribution
        - Top axis shows percentile values

        For categorical features:
        - Box plot shows distribution of predictions per category
        """
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
        except ImportError:
            raise Exception("You need to install seabor for using this functionnality")
        assert (
            percentage_ice <= 1.0 and percentage_ice >= 0.0
        ), "percentage of ice need to be 0 and 1"
        self._check_importance()

        rng = check_random_state(random_state)
        # subsample ice
        ice_lines_idx = rng.choice(
            len(self.ices_[feature_id][0]),
            int(len(self.ices_[feature_id][0]) * percentage_ice),
            replace=False,
        )
        # centers = np.array(self.values_[1:] + self.values_[:-1]) / 2
        if ax is None:
            fig, ax = plt.subplots()
        # add the percentage of the quantiles distributions
        if not self.is_categorical[feature_id]:
            # plot ice
            ax.plot(
                np.array(self.values_[feature_id]),
                np.array(self.ices_[feature_id])[:, ice_lines_idx],
                color="lightblue",
                alpha=0.5,
                linewidth=0.5,
            )
            # plot pdp
            ax.plot(
                self.values_[feature_id],
                np.average(
                    self.ices_[feature_id],
                    axis=1,
                    weights=self.sample_weight,
                ),
                color="black",
                **kwargs,
            )
            if X is not None:
                data = (_safe_indexing(X, self.features_indices[feature_id], axis=1),)
                _ax_quantiles(
                    ax,
                    np.unique(
                        np.quantile(data, np.linspace(0, 1, nbins), method="lower")
                    ),
                )
                # add distribution of value
                sns.rugplot(data, ax=ax, alpha=0.2, legend=False)
        else:
            ax.boxplot(self.ices_[feature_id])
            ax.set_xticks(np.arange(len(self.values_[feature_id])) + 1)
            ax.set_xticklabels(self.values_[feature_id])
        ax.set_xlabel(self.feature_names[feature_id])
        ax.grid(True, linestyle="-", alpha=0.4)
        return ax


def _ax_quantiles(ax, quantiles, twin="x"):
    """
    Add quantile percentage labels on a twin axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib axis to add quantile labels to.
    quantiles : array-like
        The quantile values to label.
    twin : {'x', 'y'}, default='x'
        Which axis to add the twin axis and labels to:
        - 'x': Add labels on top x-axis
        - 'y': Add labels on right y-axis

    Raises
    ------
    ValueError
        If twin is not 'x' or 'y'.

    Notes
    -----
    Creates a twin axis with percentage labels (0-100%) corresponding to the
    quantile values. Useful for showing the distribution of data alongside
    the actual values.
    """
    if twin not in ("x", "y"):
        raise ValueError("'twin' should be one of 'x' or 'y'.")

    # Duplicate the 'opposite' axis so we can define a distinct set of ticks for the
    # desired axis (`twin`).
    ax_mod = ax.twiny() if twin == "x" else ax.twinx()

    # Set the new axis' ticks for the desired axis.
    getattr(ax_mod, "set_{twin}ticks".format(twin=twin))(quantiles)
    # Set the corresponding tick labels.

    # Calculate tick label percentage values for each quantile (bin edge).
    percentages = (
        100 * np.arange(len(quantiles), dtype=np.float64) / (len(quantiles) - 1)
    )

    # If there is a fractional part, add a decimal place to show (part of) it.
    fractional = (~np.isclose(percentages % 1, 0)).astype("int8")

    getattr(ax_mod, "set_{twin}ticklabels".format(twin=twin))(
        [
            "{0:0.{1}f}%".format(percent, format_fraction)
            for percent, format_fraction in zip(percentages, fractional)
        ],
        color="#545454",
        fontsize=7,
    )
    getattr(ax_mod, "set_{twin}lim".format(twin=twin))(
        getattr(ax, "get_{twin}lim".format(twin=twin))()
    )
