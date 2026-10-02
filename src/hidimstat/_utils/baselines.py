import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class _LOCIBaselineClassifier(ClassifierMixin, BaseEstimator):
    """
    Classifier that computes the baseline predictions for LOCI,
    which is the marginal distribution of the input data.
    """

    def __init__(self):
        super().__init__()

    def fit(
        self,
        X,  # noqa: ARG002
        y,
    ):
        """
        Fit the data to compute its marginal distribution.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.

        y: array-like of shape (n_samples,)
            The input groundtruth.

        Returns
        -------
        self: :class:`hidimstat._utils.baselines._LOCIBaselineClassifier`
            The fitted classifier.
        """
        self.y_values_, self.y_counts_ = np.unique(y, return_counts=True)
        # Sklearn scorer API compatibility needs to know nb of classes
        self.classes_ = self.y_values_
        # We take the marginal probability in any case.
        # Binary classification, shape of y is (n_samples,)
        if len(self.y_values_) == 2:
            self.baseline_mean_ = self.y_counts_[1] / y.shape[0]
        # For multiclass classification, shape of y is (n_samples, n_classes)
        else:
            self.baseline_mean_ = self.y_counts_ / y.shape[0]
        return self

    def predict(self, X):
        """
        Make prediction from input data.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        out: array-like of shape (n_samples,)
            The predicted class for each sample.
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        """
        Make prediction from input data.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        out: array-like of shape (n_samples, n_classes)
            The marginal distribution of fitted data.
        """
        y_baseline = np.full(
            (X.shape[0], len(self.y_values_)), self.baseline_mean_
        )
        return y_baseline

    def decision_function(self, X):
        """
        Make prediction from input data.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        out: array-like of shape (n_samples, n_features)
            The marginal distribution of fitted data.
        """
        return self.predict_proba(X)


class _LOCIBaselineRegressor(RegressorMixin, BaseEstimator):
    """
    Classifier that computes the baseline predictions for LOCI,
    which is the marginal distribution of the input data.
    """

    def __init__(self):
        super().__init__()

    def fit(
        self,
        X,  # noqa: ARG002
        y,
    ):
        """
        Fit the data to compute its marginal distribution.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.

        y: array-like of shape (n_samples,)
            The input groundtruth.

        Returns
        -------
        self: :class:`hidimstat._utils.baselines._LOCIBaselineClassifier`
            The fitted classifier.
        """
        self.baseline_mean_ = np.mean(y)
        return self

    def predict(self, X):
        """
        Make prediction from input data.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        out: array-like of shape (n_samples, n_features)
            The marginal distribution of fitted data.
        """
        y_pred = np.full((X.shape[0],), self.baseline_mean_, dtype=float)
        return y_pred
