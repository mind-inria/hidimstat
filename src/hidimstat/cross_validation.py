from functools import partial

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold

from hidimstat.base_variable_importance import BaseVariableImportance
from hidimstat.statistical_tools.aggregation import quantile_aggregation


class VariableImportanceCrossValidation(BaseVariableImportance):
    """
    Cross validation wrapper for feature importance estimation.

    This class implements a cross validation for feature importance estimation methods.
    It splits the data into folds and fits the feature importance method on each fold,
    then aggregates the results across folds.

    Parameters
    ----------
    feature_importance : object
        Feature importance method
    cv : cross-validation generator, default=KFold()
        Determines the cross-validation splitting strategy.
    list_parameters : list of dict, optional
        List of parameter dictionaries for each fold. If provided, must match the number
        of folds in cv. Each dictionary contains parameters to initialize feature_importance
        for that specific fold.

    Attributes
    ----------
    list_feature_importance_ : list
        List of fitted feature importance methods for each fold.
    importances_ : ndarray
        Mean feature importance scores across all folds.
    pvalues_ : ndarray
        Aggregated p-values across all folds using quantile aggregation.

    Methods
    -------
    fit(X, y)
        Fit the model using X as training data and y as target values.
    importance(X, y)
        Calculate feature importance scores for X with target y.
    fit_transform(X, y)
        Fit to data, then transform it.
    """

    def __init__(
        self,
        feature_importance,
        cv=KFold(),
        list_parameters=None,
        importance_aggregation=partial(np.mean, axis=0),
        pvalue_aggregation=quantile_aggregation,
    ):
        super().__init__()
        self.feature_importance = feature_importance
        self.cv = cv
        if list_parameters is not None:
            assert (
                len(list_parameters) >= cv.get_n_splits()
            ), "the number of split of the cv should be lower than the number of list of parameters"
            for parameters in list_parameters:
                assert isinstance(
                    parameters, dict
                ), "the parameters need to be a dicstionnary"
        self.list_parameters = list_parameters
        self.importance_aggregation = importance_aggregation
        self.pvalues_aggregation = pvalue_aggregation

        self.list_feature_importance_ = None

    def fit(self, X, y):
        """
        Fit the cross-validation model.

        This method performs cross-validation by fitting the feature importance estimator
        on different training folds of the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.

        Notes
        -----
        The method creates multiple feature importance estimators, one for each fold
        of the cross-validation, and stores them in the `list_feature_importance_` attribute.
        If list_parameters is provided, each estimator is configured with the corresponding
        parameters for that fold.
        """
        self.list_feature_importance_ = []
        for index, (train, test) in enumerate(self.cv.split(X)):
            feature_importance = clone(self.feature_importance)
            if self.list_parameters is not None:
                feature_importance(**self.list_parameters[index])
            feature_importance.fit(X[train], y[train])
            self.list_feature_importance_.append(feature_importance)
        return self

    def _check_fit(self):
        """
        Checks if feature importances were already calculated.

        This internal method verifies that feature importances have been calculated before allowing
        operations that depend on them. It first checks if importances exist, then calls the parent
        class's importance check.

        Returns
        -------
        bool
            Result of parent class importance check if importances exist

        Raises
        ------
        ValueError
            If feature importances have not been calculated yet (self.list_feature_importance_ is None)
        """
        if self.list_feature_importance_ is None:
            raise ValueError(
                "The importances need to be called before calling this method"
            )

    def importance(self, X, y):
        """Calculate feature importance scores and p-values across cross-validation folds.

        This method computes feature importance scores for each fold in the cross-validation
        and aggregates them to produce final importance scores and p-values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        ndarray of shape (n_features,)
            Averaged feature importance scores across all folds.

        Attributes
        ----------
        importances_ : ndarray of shape (n_features,)
            Mean feature importance scores across all folds.
        pvalues_ : ndarray of shape (n_features,)
            Aggregated p-values across all folds using quantile aggregation.

        Notes
        -----
        This method requires that feature importance calculation has been properly
        initialized through prior method calls.
        """
        self._check_fit()
        for index, (feature_importance, (train, test)) in enumerate(
            zip(
                self.list_feature_importance_,
                self.cv.split(X),
            )
        ):
            feature_importance.importance(X[test], y[test])
        self.importances_ = self.importance_aggregation(
            [fi.importances_ for fi in self.list_feature_importance_]
        )
        self.pvalues_ = self.pvalues_aggregation(
            np.array([fi.pvalues_ for fi in self.list_feature_importance_])
        )
        return self.importances_

    def fit_importance(self, X, y):
        """
        Fits the model and returns feature importance scores.

        This method combines the functionality of `fit` and `importance` by first fitting
        the model to the data and then calculating feature importance scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Must fulfill input requirements of concrete base model.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        importance_scores : ndarray
            Feature importance scores as calculated by the `importance` method.

        See Also
        --------
        fit : Fits the model to training data
        importance : Calculates feature importance scores

        Notes
        -----
        This is a convenience method that combines model fitting and importance calculation
        in a single call. It is equivalent to calling `fit` followed by `importance`.
        """
        self.fit(X, y)
        return self.importance(X, y)
