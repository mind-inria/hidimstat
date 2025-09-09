import numpy as np
from sklearn.base import MultiOutputMixin, check_is_fitted, BaseEstimator
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from hidimstat._utils.utils import check_random_state


def _check_data_type(
    data_type: str, y: np.ndarray, categorical_max_cardinality: int
) -> str:
    """
    Check the data type and return the corresponding type. If `data_type` is "auto",
    the type is inferred from the cardinality and the type of the variable.

    Parameters
    ----------
    data_type : str
        The variable type. Supported types include "auto", "continuous", and
        "categorical".
    y : ndarray
        The group of variables to predict.
    categorical_max_cardinality : int
        The maximum cardinality of a numerical variable to be considered as categorical.

    Returns
    -------
    data_type : str
        The variable type.
    """
    if data_type in ["continuous", "categorical"]:
        return data_type
    elif data_type == "auto":
        if len(np.unique(y)) <= categorical_max_cardinality or (
            y.dtype.type is np.str_
        ):
            return "categorical"
        else:
            return "continuous"
    else:
        raise ValueError(f"type of data '{data_type}' unknow.")


class ConditionalSampler:
    def __init__(
        self,
        model_regression=None,
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
        model_categorical : sklearn compatible estimator, optional
            The model to use for categorical data. Binary is considered as a special
            case of categorical data.
        data_type : str, default="auto"
            The variable type. Supported types include "auto", "continuous", and
            "categorical". If "auto", the type is inferred from the cardinality
            of the unique values passed to the `fit` method.
        random_state : int, optional
            The random state to use for sampling.
        categorical_max_cardinality : int, default=10
            The maximum cardinality of a variable to be considered as categorical
            when `data_type` is "auto".

        """
        # check the validity of the inputs
        assert model_regression is None or issubclass(
            model_regression.__class__, BaseEstimator
        ), "Regression model invalid"
        assert model_categorical is None or issubclass(
            model_categorical.__class__, BaseEstimator
        ), "Categorial model invalid"
        self.data_type = data_type
        self.model_regression = model_regression
        self.model_categorical = model_categorical
        self.random_state = random_state
        self.categorical_max_cardinality = categorical_max_cardinality

    def fit(self, X: np.ndarray, y: np.ndarray):
        r"""
        Fit the model that estimates $\mathbb{E}[y | X]$.

        Parameters
        ----------
        X : ndarray
            The variables used to predict the group of variables $y$.
        y : ndarray
            The group of variables to predict.
        """
        self.data_type = _check_data_type(
            self.data_type, y, self.categorical_max_cardinality
        )
        self.model = (
            self.model_categorical
            if self.data_type == "categorical"
            else self.model_regression
        )

        # Group of variables
        if (y.ndim > 1) and (y.shape[1] > 1):
            if self.data_type == "categorical":
                self.model = MultiOutputClassifier(self.model)
            elif self.data_type == "continuous" and not issubclass(
                self.model.__class__, MultiOutputMixin
            ):
                self.model = MultiOutputRegressor(self.model)
            self.multioutput_ = True
        else:
            if y.ndim > 1:
                y = y.ravel()
            self.multioutput_ = False
        if self.model is None:
            raise AttributeError(f"No model was provided for {self.data_type} data")
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
        y_conditional : ndarray
            The samples from the conditional distribution.
        """
        rng = check_random_state(self.random_state)

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
                [rng.permutation(residual) for _ in range(n_samples)],
                axis=0,
            )
            return y_hat[np.newaxis, ...] + residual_permuted

        elif self.data_type == "categorical":
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
                            rng.choice(classes, p=p, size=n_samples)
                            for p in y_pred_proba[index]
                        ],
                        axis=1,
                    )
                )
            return np.stack(
                y_pred_cond,
                axis=-1,
            )
