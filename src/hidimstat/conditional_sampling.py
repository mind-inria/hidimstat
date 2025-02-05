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

        """
        self.model = model
        self.data_type = data_type
        self.model_regression = model_regression
        self.model_classification = model_classification
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.data_type == "auto":
            if (self.model_classification is None) or (self.model_regression is None):
                raise ValueError(
                    "The `model_classification` and `model_regression` attributes must \
                    be set if `data_type` is 'auto'."
                )
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
        if (self.data_type == "binary") and (not hasattr(self.model, "predict_proba")):
            raise AttributeError(
                "The model must have a `predict_proba` method to be used for binary data."
            )

        if self.data_type == "continuous":
            y_hat = self.model.predict(X)
            residual = y - y_hat
            residual_permuted = np.stack(
                [self.rng.permutation(residual) for _ in range(n_samples)],
                axis=0,
            )
            return y_hat[np.newaxis, ...] + residual_permuted

        elif self.data_type == "binary":
            y_pred_proba = self.model.predict_proba(X)

            # multioutput case (group of variables)
            if isinstance(self.model.classes_, list):
                y_pred_cond = np.stack(
                    [
                        np.stack(
                            [
                                self.rng.choice(classes, p=p, size=n_samples)
                                for p in y_proba_
                            ],
                            axis=1,
                        )
                        for y_proba_, classes in zip(y_pred_proba, self.model.classes_)
                    ],
                    axis=-1,
                )
            else:
                y_pred_cond = np.stack(
                    [
                        self.rng.choice(self.model.classes_, p=p, size=n_samples)
                        for p in y_pred_proba
                    ],
                    axis=1,
                )
            return y_pred_cond
