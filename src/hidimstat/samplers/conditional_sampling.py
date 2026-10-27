import numpy as np
from sklearn.base import (
    BaseEstimator,
    MultiOutputMixin,
    check_is_fitted,
    clone,
)
from sklearn.metrics import r2_score, roc_auc_score
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
        raise ValueError(f"type of data '{data_type}' unknown.")


class ConditionalSampler:
    def __init__(
        self,
        model_regression=None,
        model_categorical=None,
        data_type: str = "auto",
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
        categorical_max_cardinality : int, default=10
            The maximum cardinality of a variable to be considered as categorical
            when `data_type` is "auto".

        """
        self.data_type = data_type
        self.model_regression = model_regression
        self.model_categorical = model_categorical
        self.categorical_max_cardinality = categorical_max_cardinality

    def clone(self):
        model_regression = self.model_regression
        if model_regression is not None:
            model_regression = clone(model_regression)

        model_categorical = self.model_categorical
        if model_categorical is not None:
            model_categorical = clone(model_categorical)

        return ConditionalSampler(
            model_regression=model_regression,
            model_categorical=model_categorical,
            data_type=self.data_type,
            categorical_max_cardinality=self.categorical_max_cardinality,
        )

    def _check_fit_attributes(self):
        """
        Perform type and value checks on attributes needed during fitting.
        """
        # check the validity of the inputs
        if self.model_regression is not None and not isinstance(
            self.model_regression, BaseEstimator
        ):
            raise TypeError("Regression model invalid.")

        if self.model_categorical is not None and not isinstance(
            self.model_categorical, BaseEstimator
        ):
            raise TypeError("Categorical model invalid.")

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
        self._check_fit_attributes()
        self.data_type_ = _check_data_type(
            self.data_type, np.array(y), self.categorical_max_cardinality
        )

        self.model = (
            self.model_categorical
            if self.data_type_ == "categorical"
            else self.model_regression
        )
        if self.model is None:
            raise AttributeError(
                f"No model was provided for {self.data_type_} data"
            )

        # Group of variables
        if (y.ndim > 1) and (y.shape[1] > 1):
            # Data types can only be "categorical" or "continuous".
            if self.data_type_ == "categorical":
                self.model = MultiOutputClassifier(self.model)
            elif self.data_type_ == "continuous" and not isinstance(
                self.model, MultiOutputMixin
            ):
                self.model = MultiOutputRegressor(self.model)
            self.multioutput_ = True
        else:
            if y.ndim > 1 and isinstance(y, np.ndarray):
                y = y.ravel()
            self.multioutput_ = False

        self.model.fit(X, y)
        return self

    def sample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int = 1,
        random_state=None,
    ) -> np.ndarray:
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
        random_state : int, default=None
            The random state to use for sampling.

        Returns
        -------
        y_conditional : ndarray
            The samples from the conditional distribution.
        """
        check_is_fitted(self.model)
        rng = check_random_state(random_state)

        if self.data_type_ == "continuous":
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

        elif self.data_type_ == "categorical":
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
            return np.stack(y_pred_cond, axis=-1)

    def feature_predictability(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Computes how well $X^j$ is predicted from $X^{-j}$.
        For continuous data, the coefficient of determination $R^2$ is used.
        For categorical data, the ROC-AUC is computed.
        For a group of variables, the scores are averaged uniformly
        over the columns.

        Parameters
        ----------
        X : ndarray
            The complementary of the considered set of variables, $X^{-j}$.
        y : ndarray
            The group of variables to predict, $X^j$.

        Returns
        -------
        score : float
            The $R^2$ (continuous) or ROC-AUC (categorical), with OVR strategy for multiclass.
        """
        check_is_fitted(self.model)
        y = np.asarray(y)

        if self.data_type_ == "continuous":
            y_hat = np.asarray(self.model.predict(X)).reshape(y.shape)
            return r2_score(y, y_hat, multioutput="uniform_average")
        else:
            y_proba = self.model.predict_proba(X)
            if not self.multioutput_:
                y_proba, classes = [y_proba], [self.model.classes_]
            else:
                classes = self.model.classes_
            y = y.reshape(len(y), -1)
            scores = []
            for k, (proba, cls) in enumerate(
                zip(y_proba, classes, strict=False)
            ):
                if len(cls) == 2:
                    # sklearn expects the positive-class column only.
                    scores.append(roc_auc_score(y[:, k], proba[:, 1]))
                else:
                    scores.append(
                        roc_auc_score(
                            y[:, k], proba, multi_class="ovr", labels=cls
                        )
                    )
            return np.mean(scores)
