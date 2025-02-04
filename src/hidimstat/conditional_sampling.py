import numpy as np
from sklearn.base import BaseEstimator, check_is_fitted


class ConditionalSampler:
    def __init__(
        self,
        model=None,
        data_type: str = "auto",
        model_regression=None,
        model_classification=None,
        random_state: int = None,
        method_classification: str = "predict_proba",
    ):
        """
        Class use to sample from the conditional distribution $p(X^j | X^{-j})$.

        Parameters
        ----------

        model : object, optional
            The model used to estimate the conditional distribution.
        data_type : str, optional, default="auto"
            The variable type. Supported types include "auto", "continuous", "binary".
            If "auto", the type is inferred from the cardinality of the unique values
            passed to the `fit` method.
        model_regression : object, optional
            Only used if `data_type` is "auto". The model used to estimate the
            conditional for continuous variables.
        model_classification : object, optional
            Only used if `data_type` is "auto". The model used to estimate the
            conditional for binary variables.
        random_state : int, optional
            The random state to use for reproducibility.
        method_classification : str, optional
            Only used for binary variables. The method to use to get the predicted
            probabilities of the classes.
            variables.
        """
        self.model = model
        self.data_type = data_type
        self.model_regression = model_regression
        self.model_classification = model_classification
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.method_classification = method_classification

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.data_type == "auto":
            if len(np.unique(y)) == 2:
                self.data_type = "binary"
                self.model = self.model_classification
            else:
                self.data_type = "continuous"
                self.model = self.model_regression

                self.model = self.model_regression
        self.model.fit(X, y)

    def sample(self, X: np.ndarray, y: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """
        Sample from the conditional distribution $p(X^j | X^{-j})$.

        Parameters
        ----------
        X : ndarray
            The complementary of the considered set of variables, $X^{-j}$.
        y : ndarray
            The group of variables to sample, $X^j$.
        n_samples : int, optional
            The number of samples to draw.

        Returns
        -------
        ndarray
            An array of shape (n_samples, y.shape[1]) containing the samples.
        """

        check_is_fitted(self, "model")

        if self.data_type == "continuous":
            y_hat = self.model.predict(X)
            residual = y - y_hat
            residual_permuted = np.stack(
                [self.rng.permutation(residual) for _ in range(n_samples)],
                axis=0,
            )
            return y_hat[np.newaxis, ...] + residual_permuted

        elif self.data_type == "binary":
            y_pred_classes = getattr(self.model, self.method_classification)(X)
            y_pred_cond = np.stack(
                [
                    self.rng.choice(self.model.classes_, p=p, size=n_samples)
                    for p in y_pred_classes
                ],
                axis=1,
            )
            return y_pred_cond
