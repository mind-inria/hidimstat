import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class _LOCIBaselineClassifier(ClassifierMixin, BaseEstimator):
    """
    Baseline classifier class for LOCI
    """

    def __init__(self):
        super().__init__()

    def fit(
        self,
        X,  # noqa: ARG002
        y,
    ):
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
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        y_baseline = np.full(
            (X.shape[0], len(self.y_values_)), self.baseline_mean_
        )
        return y_baseline

    def decision_function(self, X):
        return self.predict_proba(X)


class _LOCIBaselineRegressor(RegressorMixin, BaseEstimator):
    """
    Baseline regressor class for LOCI
    """

    def __init__(self):
        super().__init__()

    def fit(
        self,
        X,  # noqa: ARG002
        y,
    ):
        self.baseline_mean_ = np.mean(y)
        return self

    def predict(self, X):
        y_pred = np.full((X.shape[0],), self.baseline_mean_, dtype=float)
        return y_pred
