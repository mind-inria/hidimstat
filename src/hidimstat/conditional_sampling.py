import numpy as np
from sklearn.base import MultiOutputMixin, check_is_fitted
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.utils.validation import check_random_state


class ConditionalSampler:
    def __init__(
        self,
        model_regression=None,
        model_binary=None,
        model_categorical=None,
        data_type: str = "auto",
        random_state=None,
        categorical_max_cardinality=10,
    ):
        """
        Class use to sample from the conditional distribution $p(X^j | X^{-j})$.

        Parameters
        ----------
        model_regression : sklearn compatible estimator, optional
            The model to use for continuous data.
        model_binary : sklearn compatible estimator, optional
            The model to use for binary data.
        model_categorical : sklearn compatible estimator, optional
            The model to use for categorical data.
        data_type : str, default="auto"
            The variable type. Supported types include "auto", "continuous", "binary",
            and "categorical". If "auto", the type is inferred from the cardinality of
            the unique values passed to the `fit` method. For categorical variables, the
            default strategy is to use a one-vs-rest classifier.
        random_state : int, optional
            The random state to use for sampling.
        categorical_max_cardinality : int, default=10
            The maximum cardinality of a variable to be considered as categorical
            when `data_type` is "auto".

        """
        self.data_type = data_type
        self.model_regression = model_regression
        self.model_binary = model_binary
        self.model_categorical = model_categorical

        if data_type == "auto":
            self.model_auto = {
                "continuous": model_regression,
                "binary": model_binary,
                "categorical": model_categorical,
            }
        elif data_type == "continuous":
            self.model = model_regression
        elif data_type == "binary":
            self.model = model_binary
        elif data_type == "categorical":
            self.model = model_categorical
        else:
            raise ValueError(f"type of data '{data_type}' unknow.")
        self.rng = check_random_state(random_state)
        self.categorical_max_cardinality = categorical_max_cardinality

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model that estimates $\mathbb{E}[y | X]$.

        Parameters
        ----------
        X : ndarray
            The variables used to predict the group of variables $y$.
        y : ndarray
            The group of variables to predict.
        """

        if self.data_type == "auto":
            if len(np.unique(y)) == 2:
                self.data_type = "binary"
                self.model = self.model_auto["binary"]
            elif len(np.unique(y)) <= self.categorical_max_cardinality:
                self.data_type = "categorical"
                self.model = self.model_auto["categorical"]
            else:
                self.data_type = "continuous"
                self.model = self.model_auto["continuous"]

        # Group of variables
        if (y.ndim > 1) and (y.shape[1] > 1):
            if self.data_type in ["binary", "categorical"]:
                self.model = MultiOutputClassifier(self.model)
            elif self.data_type == "continuous" and not issubclass(
                self.model.__class__, MultiOutputMixin
            ):
                self.model = MultiOutputRegressor(self.model)
            self.multioutput_ = True
        else:
            self.multioutput_ = False
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

        check_is_fitted(self.model)

        if self.data_type == "continuous":
            if not hasattr(self.model, "predict"):
                raise AttributeError(
                    "The model must have a `predict` method to be used for \
                        continuous data."
                )
            y_hat = self.model.predict(X).reshape(y.shape)
            residual = y - y_hat
            residual_permuted = np.stack(
                [self.rng.permutation(residual) for _ in range(n_samples)],
                axis=0,
            )
            return y_hat[np.newaxis, ...] + residual_permuted

        elif self.data_type in ["binary", "categorical"]:
            if not hasattr(self.model, "predict_proba"):
                raise AttributeError(
                    "The model must have a `predict_proba` method to be used for \
                        categorical or binary data."
                )
            y_pred_proba = self.model.predict_proba(X)

            # multioutput case (group of variables)
            if not self.multioutput_:
                y_classes = [self.model.classes_]
                y_pred_proba = [y_pred_proba]
            else:
                y_classes = self.model.classes_

            y_pred_cond = []
            for index, classes in enumerate(y_classes):
                y_pred_cond.append(
                    np.stack(
                        [
                            self.rng.choice(classes, p=p, size=n_samples)
                            for p in y_pred_proba[index]
                        ],
                        axis=1,
                    )
                )
            return np.stack(
                y_pred_cond,
                axis=-1,
            )
