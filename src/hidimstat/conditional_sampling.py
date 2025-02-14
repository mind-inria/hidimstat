import numpy as np
from sklearn.base import check_is_fitted
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.utils.validation import check_random_state


class ConditionalSampler:
    def __init__(
        self,
        model_regression=None,
        model_binary=None,
        model_categorical=None,
        model_ordinary=None,
        data_type: str = "auto",
        imputation_model_multimodal= False,
        random_state=None,
        categorical_max_cardinality=10,
    ):
        """
        Class use to sample from the conditional distribution $p(X^j | X^{-j})$.

        Parameters
        ----------

        model : object
            The model used to estimate the conditional distribution.
        data_type : str, default="auto"
            The variable type. Supported types include "auto", "continuous", "binary",
            and "categorical". If "auto", the type is inferred from the cardinality of
            the unique values passed to the `fit` method. For categorical variables, the
            default strategy is to use a one-vs-rest classifier.
        model_regression : object
            Only used if `data_type` is "auto". The model used to estimate the
            conditional for continuous variables.
        model_classification : object
            Only used if `data_type` is "auto". The model used to estimate the
            conditional for binary variables.
        random_state : int
            The random state to use for sampling.
        categorical_max_cardinality : int
            The maximum cardinality of a variable to be considered as categorical
            when `data_type` is "auto".

        """
        self.data_type = data_type
        if not isinstance(imputation_model_multimodal, list):
            imputation_model_multimodal = [imputation_model_multimodal for _ in range(4)]
        else:
            imputation_model_multimodal = imputation_model_multimodal
        if data_type == "auto":
            self.model = (model_regression, model_binary, model_categorical)
            self.imputation_model_multimodal = imputation_model_multimodal
        elif data_type == "continuous":
            self.model = model_regression
            self.imputation_model_multimodal = imputation_model_multimodal[0]
        elif data_type == "binary":
            self.model = model_binary
            self.imputation_model_multimodal = imputation_model_multimodal[1]
        elif data_type == "categorical":
            self.model = model_categorical
            self.imputation_model_multimodal = imputation_model_multimodal[2]
        elif data_type == "ordinal":
            self.model = model_ordinary
            self.imputation_model_multimodal = imputation_model_multimodal[3]
        else:
            raise ValueError(f"type of data '{data_type}' unknow.")
        self.rng = check_random_state(random_state)
        self.categorical_max_cardinality = categorical_max_cardinality


    def fit(self, X: np.ndarray, y: np.ndarray):

        if self.data_type == "auto":
            if len(np.unique(y)) == 2:
                self.data_type = "binary"
                self.model = self.model[1]
                self.imputation_model_multimodal = self.imputation_model_multimodal[1]
            elif len(np.unique(y)) <= self.categorical_max_cardinality:
                self.data_type = "categorical"
                self.model = self.model[2]
                self.imputation_model_multimodal = self.imputation_model_multimodal[2]
            else:
                self.data_type = "continuous"
                self.model = self.model[0]
                self.imputation_model_multimodal = self.imputation_model_multimodal[0]

        # Group of variables
        if ((y.ndim > 1) and (y.shape[1] > 1) and not self.imputation_model_multimodal):
            if (self.data_type in ["binary", "categorical"]):
                self.model = MultiOutputClassifier(self.model)
            else:
                self.model = MultiOutputRegressor(self.model)
            self.imputation_model_multimodal = True

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
        if (self.data_type in ["binary", "categorical"]) and (not hasattr(self.model, "predict_proba")):
            raise AttributeError(
                "The model must have a `predict_proba` method to be used for categorical or binary data."
            )

        if self.data_type in ["continuous", "ordinal"]:
            y_hat = self.model.predict(X)
            residual = y - y_hat
            residual_permuted = np.stack(
                [self.rng.permutation(residual) for _ in range(n_samples)],
                axis=0,
            )
            return y_hat[np.newaxis, ...] + residual_permuted

        elif self.data_type in ["binary", "categorical"]:
            y_pred_proba = self.model.predict_proba(X)

            # multioutput case (group of variables)
            if not self.imputation_model_multimodal:
                y_classes = [self.model.classes_]
                y_pred_proba = [y_pred_proba]
            else:
                y_classes = self.model.classes_

            y_pred_cond = []
            for index, classes in enumerate(y_classes):
                    y_pred_cond.append(np.stack([
                        self.rng.choice(classes, p=p, size=n_samples)
                        for p in y_pred_proba[index]
                    ], axis=1))
            return np.stack(y_pred_cond,axis=-1,)
