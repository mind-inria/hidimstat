from collections.abc import Iterable
from typing import override
import warnings

from joblib import Parallel, delayed
import numpy as np
from scipy import sparse
from scipy.stats.mstats import mquantiles
from sklearn.base import is_classifier, is_regressor
from sklearn.utils import _safe_indexing, check_array
from sklearn.utils._indexing import (
    _determine_key_type,
    _get_column_indices,
    _safe_assign,
)
from sklearn.utils._response import _get_response_values
from sklearn.utils.validation import _check_sample_weight, check_is_fitted
from sklearn.utils.extmath import cartesian
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
    """Generate a grid of points based on the percentiles of X.

    The grid is a cartesian product between the columns of ``values``. The
    ith column of ``values`` consists in ``grid_resolution`` equally-spaced
    quantile of the distribution of jth column of X.

    If ``grid_resolution`` is bigger than the number of unique values in the
    j-th column of X or if the feature is a categorical feature (by inspecting
    `is_categorical`) , then those unique values will be used instead.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_target_features)
        The data.

    is_categorical : list of bool
        For each feature, tells whether it is categorical or not. If a feature
        is categorical, then the values used will be the unique ones
        (i.e. categories) instead of the percentiles.

    grid_resolution : int
        The number of equally spaced points to be placed on the grid for each
        feature.

    custom_values: dict
        Mapping from column index of X to an array-like of values where
        the partial dependence should be calculated for that feature

    Returns
    -------
    grid : ndarray of shape (n_points, n_target_features)
        A value for each feature at each point in the grid. ``n_points`` is
        always ``<= grid_resolution ** X.shape[1]``.

    values : list of 1d ndarrays
        The values with which the grid has been created. The size of each
        array ``values[j]`` is either ``grid_resolution``, the number of
        unique values in ``X[:, j]``, if j is not in ``custom_range``.
        If j is in ``custom_range``, then it is the length of ``custom_range[j]``.
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
        if feature_idx in custom_values:
            # Use values in the custom range
            axis = custom_values[feature_idx]
        else:
            try:
                uniques = np.unique(_safe_indexing(X, feature_idx, axis=1))
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
                            _safe_indexing(X, feature_idx, axis=1),
                            prob=np.linspace(0.0, 1.0, grid_resolution + 1),
                            axis=0,
                        )
                    )
                else:
                    emp_percentiles = mquantiles(
                        _safe_indexing(X, feature_idx, axis=1), prob=percentiles, axis=0
                    )
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
        dtype = _safe_indexing(X, feature_idx, axis=1).dtype
        if dtype in np._core._type_aliases.allTypes.values() and (
            np.isdtype(dtype, "bool")
            or np.isdtype(dtype, "integral")
            or np.isdtype(dtype, "numeric")
        ):
            digitize = (
                np.digitize(_safe_indexing(X, feature_idx, axis=1), axis, right=True)
                - 1
            )  # correction of the number of classes
        else:
            string_to_num = {
                s: i
                for i, s in enumerate(
                    sorted(set(_safe_indexing(X, feature_idx, axis=1)))
                )
            }
            digitize = np.digitize(
                np.array(
                    [string_to_num[s] for s in _safe_indexing(X, feature_idx, axis=1)]
                ),
                [string_to_num[s] for s in axis],
                right=True,
            )
        indexes.append(np.clip(digitize, 0, None))

        print(values[-1], values[-1].shape, np.unique(axis).shape)
        print(indexes[-1], indexes[-1].shape, np.unique(indexes[-1]).shape)
    return values, indexes


def _partial_dependence_brute(est, grid, features, X, method, cross_features, n_jobs):
    """Calculate partial dependence via the brute force method.

    The brute method explicitly averages the predictions of an estimator over a
    grid of feature values.

    For each `grid` value, all the samples from `X` have their variables of
    interest replaced by that specific `grid` value. The predictions are then made
    and averaged across the samples.

    This method is slower than the `'recursion'`
    (:func:`~sklearn.inspection._partial_dependence._partial_dependence_recursion`)
    version for estimators with this second option. However, with the `'brute'`
    force method, the average will be done with the given `X` and not the `X`
    used during training, as it is done in the `'recursion'` version. Therefore
    the average can always accept `sample_weight` (even when the estimator was
    fitted without).

    Parameters
    ----------
    est : BaseEstimator
        A fitted estimator object implementing :term:`predict`,
        :term:`predict_proba`, or :term:`decision_function`.
        Multioutput-multiclass classifiers are not supported.

    grid : array-like of shape (n_points, n_target_features)
        The grid of feature values for which the partial dependence is calculated.
        Note that `n_points` is the number of points in the grid and `n_target_features`
        is the number of features you are doing partial dependence at.

    features : array-like of {int, str}
        The feature (e.g. `[0]`) or pair of interacting features
        (e.g. `[(0, 1)]`) for which the partial dependency should be computed.

    X : array-like of shape (n_samples, n_features)
        `X` is used to generate values for the complement features. That is, for
        each value in `grid`, the method will average the prediction of each
        sample from `X` having that grid value for `features`.

    method : {'auto', 'predict_proba', 'decision_function'}, \
            default='auto'
        Specifies whether to use :term:`predict_proba` or
        :term:`decision_function` as the target response. For regressors
        this parameter is ignored and the response is always the output of
        :term:`predict`. By default, :term:`predict_proba` is tried first
        and we revert to :term:`decision_function` if it doesn't exist.

    Returns
    -------
    predictions : array-like
        The predictions for the given `grid` of features values over the samples
        from `X`. For non-multioutput regression and binary classification the
        shape is `(n_instances, n_points)` and for multi-output regression and
        multiclass classification the shape is `(n_targets, n_instances, n_points)`,
        where `n_targets` is the number of targets (`n_tasks` for multi-output
        regression, and `n_classes` for multiclass classification), `n_instances`
        is the number of instances in `X`, and `n_points` is the number of points
        in the `grid`.
    """
    predictions = []

    if method == "auto":
        method = (
            "predict" if is_regressor(est) else ["predict_proba", "decision_function"]
        )
    if cross_features:
        predictions = Parallel(n_jobs=n_jobs)(
            delayed(_joblib_get_predictions_cross)(new_values, features, X, est, method)
            for new_values in cartesian(grid)
        )
        n_samples = X.shape[0]

        # reshape to (n_targets, n_instances, n_points) where n_targets is:
        # - 1 for non-multioutput regression and binary classification (shape is
        #   already correct in those cases)
        # - n_tasks for multi-output regression
        # - n_classes for multiclass classification.
        predictions = np.array(predictions).T
        if is_regressor(est) and predictions.ndim == 2:
            # non-multioutput regression, shape is (n_instances, n_points,)
            predictions = predictions.reshape(n_samples, -1)
        elif is_classifier(est) and predictions.shape[0] == 2:
            # Binary classification, shape is (2, n_instances, n_points).
            # we output the effect of **positive** class
            predictions = predictions[1]
            predictions = predictions.reshape(n_samples, -1)

        # reshape predictions to
        # (n_outputs, n_instances, n_values_feature_0, n_values_feature_1, ...)
        predictions = predictions.reshape(
            -1, n_samples, *[val.shape[0] for val in grid]
        )
    else:
        predictions = Parallel(n_jobs=n_jobs)(
            delayed(_joblib_get_predictions)(variable, values, X, est, method)
            for variable, values in zip(features, grid)
        )
    return predictions


def _joblib_get_predictions(variable, values, X, est, method):
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

        predictions.append(pred)
    return predictions


def _joblib_get_predictions_cross(new_values, features, X, est, method):
    predictions = []
    for i, variable in enumerate(features):
        _safe_assign(X, new_values[i], column_indexer=variable)

        # Note: predictions is of shape
        # (n_points,) for non-multioutput regressors
        # (n_points, n_tasks) for multioutput regressors
        # (n_points, 1) for the regressors in cross_decomposition (I think)
        # (n_points, 1) for binary classification (positive class already selected)
        # (n_points, n_classes) for multiclass classification
        pred, _ = _get_response_values(est, X, response_method=method)

        predictions.append(pred)
    return predictions


class PartialDependancePlot(BaseVariableImportance):
    def __init__(
        self,
        estimator,
        features,
        method: str = "auto",
        n_jobs: int = 1,
        sample_weight=None,
        categorical_features=None,
        feature_names=None,
        percentiles=(0.05, 0.95),
        grid_resolution=100,
        custom_values=None,
        resolution_statistique=False,
        cross_features=False,
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

        if is_regressor(self.estimator) and method != "auto":
            raise ValueError(
                "The method parameter is ignored for regressors and " "must be 'auto'."
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
        self.cross_features = cross_features

    @override
    def fit(self, X=None, y=None):
        if X is not None:
            warnings.warn("X won't be used")
        if y is not None:
            warnings.warn("y won't be used")
        return self

    def importance(self, X, y=None):
        if y is not None:
            warnings.warn("y won't be used")

        # Use check_array only on lists and other non-array-likes / sparse. Do not
        # convert DataFrame into a NumPy array.
        if not (hasattr(X, "__array__") or sparse.issparse(X)):
            X_ = check_array(X, ensure_all_finite="allow-nan", dtype=object)
        else:
            X_ = X

        if self.sample_weight is not None:
            self.sample_weight = _check_sample_weight(self.sample_weight, X)

        if _determine_key_type(self.features, accept_slice=False) == "int":
            # _get_column_indices() supports negative indexing. Here, we limit
            # the indexing to be positive. The upper bound will be checked
            # by _get_column_indices()
            if np.any(np.less(self.features, 0)):
                raise ValueError(
                    "all features must be in [0, {}]".format(X.shape[1] - 1)
                )

        features_indices = np.asarray(
            _get_column_indices(X_, self.features), dtype=np.intp, order="C"
        ).ravel()

        feature_names = _check_feature_names(X_, self.feature_names)

        n_features = X_.shape[1]
        if self.categorical_features is None:
            is_categorical = [False] * len(features_indices)
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
                is_categorical = [categorical_features[idx] for idx in features_indices]
            elif categorical_features.dtype.kind in ("i", "O", "U"):
                # categorical features provided as a list of indices or feature names
                categorical_features_idx = [
                    _get_feature_index(cat, feature_names=feature_names)
                    for cat in categorical_features
                ]
                is_categorical = [
                    idx in categorical_features_idx for idx in features_indices
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
            features_indices, self.features, is_categorical
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

        X_subset = _safe_indexing(X_, features_indices, axis=1)

        custom_values_for_X_subset = {
            index: custom_values.get(feature)
            for index, feature in enumerate(self.features)
            if feature in custom_values
        }
        self.values_, _ = _grid_from_X(
            X_subset,
            self.percentiles,
            is_categorical,
            self.grid_resolution,
            custom_values_for_X_subset,
            self.resolution_statistique,
        )

        self.ices_ = _partial_dependence_brute(
            self.estimator,
            self.values_,
            features_indices,
            X_,
            self.method,
            self.cross_features,
            self.n_jobs,
        )

        averaged_predictions = []
        for ice in self.ices_:
            # average over samples
            averaged_predictions.append(
                np.average(ice, axis=0, weights=self.sample_weight)
            )
        # reshape averaged_predictions to (n_targets, n_points) where n_targets is:
        # - 1 for non-multioutput regression and binary classification (shape is
        #   already correct in those cases)
        # - n_tasks for multi-output regression
        # - n_classes for multiclass classification.
        averaged_predictions = np.array(averaged_predictions).T
        if averaged_predictions.ndim == 1:
            # reshape to (1, n_points) for consistency with
            # _partial_dependence_recursion
            averaged_predictions = averaged_predictions.reshape(1, -1)
        if self.cross_features:
            # reshape averaged_predictions to
            # (n_outputs, n_values_feature_0, n_values_feature_1, ...)
            averaged_predictions = averaged_predictions.reshape(
                -1, *[val.shape[0] for val in self.values_]
            )

        self.importances_ = np.mean(averaged_predictions, axis=0)
        self.pvalues_ = None
        return self.importances_

    def fit_importance(self, X, y=None, cv=None):
        if y is not None:
            warnings.warn("y won't be used")
        if cv is not None:
            warnings.warn("cv won't be used")
        self.fit()
        return self.importance(X)

    def plot(self, feature_id, ax=None, X=None, **kwargs):
        """
        base on https://github.com/blent-ai/ALEPython/blob/dev/src/alepython/ale.py#L159

        Parameters
        ----------
        feature_id : _type_
            _description_
        ax : _type_
            _description_
        X : _type_, optional
            _description_, by default None

        Raises
        ------
        Exception
            _description_
        """
        self._check_importance()
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
        except ImportError:
            raise Exception("You need to install seabor for using this functionnality")
        # centers = np.array(self.values_[1:] + self.values_[:-1]) / 2
        if ax is None:
            fig, ax = plt.subplots()
        # plot pdp
        ax.plot(
            self.values_[feature_id],
            np.average(self.ices_[feature_id], axis=1, weights=self.sample_weight),
            **kwargs,
        )
        # plot ice
        # TODO need to be in ligh grey
        # ax.plot(cartesian(self.values_), self.ices_[feature_id])
        ax.grid(True, linestyle="-", alpha=0.4)
        # add distribution of value
        if X is not None:
            sns.rugplot(X[feature_id], ax=ax, alpha=0.2)
        # add the percentage of the quantiles distributions
        dtype = type(self.values_[feature_id][0])
        if dtype in np._core._type_aliases.allTypes.values() and (
            np.isdtype(dtype, "bool")
            or np.isdtype(dtype, "integral")
            or np.isdtype(dtype, "numeric")
        ):
            _ax_quantiles(ax, self.values_[feature_id])
        return ax


def _ax_quantiles(ax, quantiles, twin="x"):
    """Plot quantiles of a feature onto axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to modify.
    quantiles : array-like
        Quantiles to plot.
    twin : {'x', 'y'}, optional
        Select the axis for which to plot quantiles.

    Raises
    ------
    ValueError
        If `twin` is not one of 'x' or 'y'.

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


if __name__ == "__main__":
    X = [[0, 0, 2], [1, 0, 0]]

    y = [0, 1]

    from sklearn.ensemble import GradientBoostingClassifier

    gb = GradientBoostingClassifier(random_state=0).fit(X, y)

    pdp = PartialDependancePlot(
        estimator=gb, features=[0], percentiles=(0, 1), grid_resolution=2
    )
    pdp.fit()
    pdp.importance(X=X)
