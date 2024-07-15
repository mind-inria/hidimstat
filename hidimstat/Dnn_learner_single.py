import itertools

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from .utils import (
    create_X_y,
    dnn_net,
    joblib_ensemble_dnnet,
    ordinal_encode,
    relu,
    sigmoid,
    softmax,
)


class Dnn_learner_single(BaseEstimator):
    """
    This class implements the Multi-Layer Perceptron (MLP) default inference
    learner for Block-Based Importance (BBI) framework.

    Parameters
    ----------
    encode : bool, default=False
        Whether to encode the categorical outcome.
    do_hypertuning : bool, default=True
        Tuning the hyperparameters of the provided estimator.
    dict_hypertuning : dict, default=None
        The dictionary of hyperparameters to tune.
    n_ensemble : int, default=10
        The number of sub-DNN models to fit to the data.
    min_keep : int, default=10
        The minimal number of sub-DNNs to keep if > 10.
    batch_size : int, default=32
        The number of samples per batch for training.
    batch_size_val : int, default=128
        The number of samples per batch for validation.
    n_epoch : int, default=200
        The number of epochs for the DNN learner(s).
    verbose : int, default=0
        If verbose > 0, the fitted iterations will be printed.
    sampling_with_repitition : bool, default=True
        Application of sampling_with_repitition sampling for the training set
    split_percentage : float, default=0.8
        The training/validation cut for the provided data.
    problem_type : str, default='regression'
        A classification or a regression problem.
    list_continuous : list, default=None
        The list of continuous variables.
    list_grps : list of lists, default=None
        A list collecting the indices of the groups' variables
        while applying the stacking method.
    beta1 : float, default=0.9
        The exponential decay rate for the first moment estimates.
    beta2 : float, default=0.999
        The exponential decay rate for the second moment estimates.
    lr : float, default=1e-3
        The learning rate.
    epsilon : float, default=1e-8
        A small constant added to the denominator to prevent division by zero.
    l1_weight : float, default=1e-2
        The L1-regularization paramter for weight decay.
    l2_weight : float, default=0
        The L2-regularization paramter for weight decay.
    n_jobs : int, default=1
        The number of workers for parallel processing.
    group_stacking : bool, default=False
        Apply the stacking-based method for the provided groups.
    input_dimensions : list, default=None
        The cumsum of inputs after the linear sub-layers.
    random_state : int, default=2023
        Fixing the seeds of the random generator.
    """

    def __init__(
        self,
        encode=False,
        do_hypertuning=False,
        dict_hypertuning=None,
        n_ensemble=10,
        min_keep=10,
        batch_size=32,
        batch_size_val=128,
        n_epoch=200,
        verbose=0,
        sampling_with_repitition=True,
        split_percentage=0.8,
        problem_type="regression",
        list_continuous=None,
        list_grps=None,
        beta1=0.9,
        beta2=0.999,
        lr=1e-3,
        epsilon=1e-8,
        l1_weight=1e-2,
        l2_weight=0,
        n_jobs=1,
        group_stacking=False,
        input_dimensions=None,
        random_state=2023,
    ):
        self.encode = encode
        self.do_hypertuning = do_hypertuning
        self.dict_hypertuning = dict_hypertuning
        self.n_ensemble = n_ensemble
        self.min_keep = min_keep
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.n_epoch = n_epoch
        self.verbose = verbose
        self.sampling_with_repitition = sampling_with_repitition
        self.split_percentage = split_percentage
        self.problem_type = problem_type
        self.list_grps = list_grps
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.epsilon = epsilon
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.list_continuous = list_continuous
        self.n_jobs = n_jobs
        self.group_stacking = group_stacking
        self.input_dimensions = input_dimensions
        self.random_state = random_state
        self.enc_y = []
        self.activation_outcome = {
            "classification": softmax,
            "ordinal": sigmoid,
            "binary": sigmoid,
        }
        self.is_encoded = False

    def fit(self, X, y=None):
        """
        Build the DNN learner with the training set (X, y)

        Parameters
        ----------
        X : {pandas dataframe, array-like, sparse matrix} of shape (n_samples,
        n_features)
            The training input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs),
        default=None
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        # Disabling the encoding parameter with the regression case
        if self.problem_type == "regression":
            self.encode = False

        if self.encode:
            y_encoded = self.encode_outcome(y)
            self.is_encoded = True
            y_encoded = np.squeeze(y_encoded, axis=0)
        else:
            y_encoded = y.copy()

        # Initializing the dictionary for tuning the hyperparameters
        if self.dict_hypertuning is None:
            self.dict_hypertuning = {
                "lr": [1e-2, 1e-3, 1e-4],
                "l1_weight": [0, 1e-2, 1e-4],
                "l2_weight": [0, 1e-2, 1e-4],
            }

        # Switch to the special binary case
        if (self.problem_type == "classification") and (y.shape[-1] < 3):
            self.problem_type = "binary"

        _, p = X.shape
        self.min_keep = max(min(self.min_keep, self.n_ensemble), 1)
        rng = np.random.RandomState(self.random_state)
        list_seeds = rng.randint(1e5, size=(self.n_ensemble))

        # Initialize the list of continuous variables
        if self.list_continuous is None:
            self.list_continuous = list(np.arange(p))

        # Initialize the list of groups
        if self.list_grps is None:
            if not self.group_stacking:
                self.list_grps = []

        # Convert the matrix of predictors to numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # Hyperparameter tuning
        if self.do_hypertuning:
            self.__tuning_hyper(X, y)

        parallel = Parallel(
            n_jobs=min(self.n_jobs, self.n_ensemble), verbose=self.verbose
        )
        res_ens = list(
            zip(
                *parallel(
                    delayed(joblib_ensemble_dnnet)(
                        X,
                        y_encoded,
                        problem_type=self.problem_type,
                        activation_outcome=self.activation_outcome,
                        list_continuous=self.list_continuous,
                        list_grps=self.list_grps,
                        sampling_with_repitition=self.sampling_with_repitition,
                        split_percentage=self.split_percentage,
                        group_stacking=self.group_stacking,
                        input_dimensions=self.input_dimensions,
                        n_epoch=self.n_epoch,
                        batch_size=self.batch_size,
                        beta1=self.beta1,
                        beta2=self.beta2,
                        lr=self.lr,
                        l1_weight=self.l1_weight,
                        l2_weight=self.l2_weight,
                        epsilon=self.epsilon,
                        random_state=list_seeds[i],
                    )
                    for i in range(self.n_ensemble)
                )
            )
        )
        pred_m = np.array(res_ens[3])
        loss = np.array(res_ens[4])

        if self.n_ensemble == 1:
            return [(res_ens[0][0], (res_ens[1][0], res_ens[2][0]))]

        # Keeping the optimal subset of DNNs
        sorted_loss = loss.copy()
        sorted_loss.sort()
        new_loss = np.empty(self.n_ensemble - 1)
        for i in range(self.n_ensemble - 1):
            current_pred = np.mean(pred_m[loss >= sorted_loss[i], :], axis=0)
            if self.problem_type == "regression":
                new_loss[i] = mean_squared_error(y, current_pred)
            else:
                new_loss[i] = log_loss(y, current_pred)
        keep_dnn = (
            loss
            >= sorted_loss[np.argmin(new_loss[: (self.n_ensemble - self.min_keep + 1)])]
        )

        self.optimal_list = [
            (res_ens[0][i], (res_ens[1][i], res_ens[2][i]))
            for i in range(self.n_ensemble)
            if keep_dnn[i]
        ]
        self.pred = [None] * len(self.optimal_list)
        self.is_fitted = True
        return self

    def encode_outcome(self, y, train=True):
        """
        This function encodes the categorical outcome

        Parameters
        ----------
        y : array-like of shaoe (n_samples,)
            The categorical outcome.
        train : bool, default=True
            Whether to fit or not the encoder.
        """
        list_y = []
        if len(y.shape) != 2:
            y = y.reshape(-1, 1)
        if self.problem_type == "regression":
            list_y.append(y)

        for col in range(y.shape[1]):
            if train:
                # Encoding the target with the classification case
                if self.problem_type in ("classification", "binary"):
                    self.enc_y.append(OneHotEncoder(handle_unknown="ignore"))
                    curr_y = self.enc_y[col].fit_transform(y[:, [col]]).toarray()
                    list_y.append(curr_y)

                # Encoding the target with the ordinal case
                if self.problem_type == "ordinal":
                    y = ordinal_encode(y)

            else:
                # Encoding the target with the classification case
                if self.problem_type in ("classification", "binary"):
                    curr_y = self.enc_y[col].transform(y[:, [col]]).toarray()
                    list_y.append(curr_y)

                ## ToDo Add the ordinal case
        return np.array(list_y)

    def hyper_tuning(
        self,
        X_train,
        y_train,
        X_valid,
        y_valid,
        list_hyper=None,
        random_state=None,
    ):
        """
        This function supports the tuning of the hyperparameters.

        Parameters
        ----------
        X_train : {array-like, sparse matrix} of shape (n_train_samples, n_features)
            The training input samples.
        y_train : array-like of shape (n_train_samples,) or (n_train_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression) for the training samples.
        X_train : {array-like, sparse matrix} of shape (n_valid_samples, n_features)
            The validation input samples.
        y_train : array-like of shape (n_valid_samples,) or (n_valid_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression) for the validation samples.
        list_hyper : list of tuples, default=None
            The list of tuples for the hyperparameters values.
        random_state : int, default=None
            Fixing the seeds of the random generator.
        """
        parallel = Parallel(
            n_jobs=min(self.n_jobs, self.n_ensemble), verbose=self.verbose
        )
        y_train = self.encode_outcome(y_train)
        y_valid = self.encode_outcome(y_valid, train=False)
        return [
            list(
                zip(
                    *parallel(
                        delayed(dnn_net)(
                            X_train,
                            y_train[i, ...],
                            X_valid,
                            y_valid[i, ...],
                            problem_type=self.problem_type,
                            n_epoch=self.n_epoch,
                            batch_size=self.batch_size,
                            beta1=self.beta1,
                            beta2=self.beta2,
                            lr=el[0],
                            l1_weight=el[1],
                            l2_weight=el[2],
                            epsilon=self.epsilon,
                            list_grps=self.list_grps,
                            group_stacking=self.group_stacking,
                            input_dimensions=self.input_dimensions,
                            random_state=random_state,
                        )
                        for el in list_hyper
                    )
                )
            )[2]
            for i in range(y_train.shape[0])
        ]

    def __tuning_hyper(self, X, y):
        """
        This function tunes the hyperparameters for the DNN learner.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_train_samples, n_features)
            The training input samples.
        y : array-like of shape (n_train_samples,) or (n_train_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        """
        (
            X_train_scaled,
            y_train_scaled,
            X_valid_scaled,
            y_valid_scaled,
            X_scaled,
            __,
            scaler_x,
            scaler_y,
            ___,
        ) = create_X_y(
            X,
            y,
            sampling_with_repitition=self.sampling_with_repitition,
            split_percentage=self.split_percentage,
            problem_type=self.problem_type,
            list_continuous=self.list_continuous,
            random_state=self.random_state,
        )
        list_hyper = list(itertools.product(*list(self.dict_hypertuning.values())))
        list_loss = self.hyper_tuning(
            X_train_scaled,
            y_train_scaled,
            X_valid_scaled,
            y_valid_scaled,
            list_hyper,
            random_state=self.random_state,
        )
        ind_min = np.argmin(list_loss)
        best_hyper = list_hyper[ind_min]
        if not isinstance(best_hyper, dict):
            best_hyper = dict(zip(self.dict_hypertuning.keys(), best_hyper))
        self.set_params(**best_hyper)

    def predict(self, X, scale=True):
        """
        This function predicts the regression target for the input samples X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_test_samples, n_features),
            default=None
            The input samples.
        scale : bool, default=True
            Whether to scale the continuous input variables.

        Returns
        -------
        res_pred : {array-like, sparse matrix)
            The average predictions across the sub-DNN models.
        """
        if self.problem_type != "regression":
            raise Exception("Use the predict_proba function for classification")

        # Prepare the test set for the prediction
        if scale:
            X = self.__scale_test(X)

        # Process the common prediction part
        self.__pred_common(X)

        res_pred = np.zeros((self.pred[0].shape))
        total_n_elements = 0
        for ind_mod, pred in enumerate(self.pred):
            res_pred += (
                pred * self.optimal_list[ind_mod][1][1].scale_
                + self.optimal_list[ind_mod][1][1].mean_
            )
            total_n_elements += 1
        res_pred = res_pred.copy() / total_n_elements

        return res_pred

    def predict_proba(self, X, scale=True):
        """
        This function predicts the class probabilities for the input samples X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_test_samples, n_features),
            default=None
            The input samples.
        scale : bool, default=True
            Whether to scale the continuous input variables.

        Returns
        -------
        res_pred : {array-like, sparse matrix)
            The average predictions across the sub-DNN models.
        """
        if self.problem_type == "regression":
            raise Exception("Use the predict function for classification")

        # Prepare the test set for the prediction
        if scale:
            X = self.__scale_test(X)

        # Process the common prediction part
        self.__pred_common(X)

        res_pred = np.zeros((self.pred[0].shape))
        total_n_elements = 0
        for pred in self.pred:
            res_pred += self.activation_outcome[self.problem_type](pred)
            total_n_elements += 1
        res_pred = res_pred.copy() / total_n_elements
        if self.problem_type == "binary":
            res_pred = np.array(
                [[1 - res_pred[i][0], res_pred[i][0]] for i in range(res_pred.shape[0])]
            )

        return res_pred

    def __scale_test(self, X):
        """
        This function prepares the input of the DNN estimator either in the
        default case or after applying the stacking method

        Parameters
        ----------
        X : {array-like, sparse-matrix} of shape (n_test_samples, n_features)
            The input samples.

        Returns
        -------
        X_test_n : {array-like, sparse-matrix}
            The new scaled matrix of the test set.
        """
        # Check is fit had been called
        check_is_fitted(self, ["is_fitted"])

        if isinstance(X, pd.DataFrame):
            X = np.array(X)

        # The input will be either the original input or the result
        # of the provided sub-linear layers in a stacking way for the different groups
        # In the stacking method, each sub-linear layer will have a corresponding output
        if self.group_stacking:
            X_test_n = [None] * len(self.optimal_list)
            for mod in range(len(self.optimal_list)):
                X_test_scaled = X.copy()
                if len(self.list_continuous) > 0:
                    X_test_scaled[:, self.list_continuous] = self.optimal_list[mod][1][
                        0
                    ].transform(X_test_scaled[:, self.list_continuous])
                X_test_n_curr = np.zeros(
                    (
                        X_test_scaled.shape[0],
                        self.input_dimensions[-1],
                    )
                )
                for grp_ind in range(len(self.list_grps)):
                    curr_pred = X_test_scaled[:, self.list_grps[grp_ind]].copy()
                    n_layer_stacking = len(self.optimal_list[mod][0][3][grp_ind]) - 1
                    for ind_w_b in range(n_layer_stacking):
                        if ind_w_b == 0:
                            curr_pred = relu(
                                X_test_scaled[:, self.list_grps[grp_ind]].dot(
                                    self.optimal_list[mod][0][3][grp_ind][ind_w_b]
                                )
                                + self.optimal_list[mod][0][4][grp_ind][ind_w_b]
                            )
                        else:
                            curr_pred = relu(
                                curr_pred.dot(
                                    self.optimal_list[mod][0][3][grp_ind][ind_w_b]
                                )
                                + self.optimal_list[mod][0][4][grp_ind][ind_w_b]
                            )
                    X_test_n_curr[
                        :,
                        list(
                            np.arange(
                                self.input_dimensions[grp_ind],
                                self.input_dimensions[grp_ind + 1],
                            )
                        ),
                    ] = (
                        curr_pred.dot(
                            self.optimal_list[mod][0][3][grp_ind][n_layer_stacking]
                        )
                        + self.optimal_list[mod][0][4][grp_ind][n_layer_stacking]
                    )
                    # X_test_n_curr[
                    #     :,
                    #     list(
                    #         np.arange(
                    #             self.input_dimensions[grp_ind],
                    #             self.input_dimensions[grp_ind + 1],
                    #         )
                    #         # np.arange(
                    #         #     self.n_out_subLayers * grp_ind,
                    #         #     (grp_ind + 1) * self.n_out_subLayers,
                    #         # )
                    #     ),
                    # ] = (
                    #     X_test_scaled[:, self.list_grps[grp_ind]].dot(
                    #         self.optimal_list[mod][0][3][grp_ind]
                    #     )
                    #     + self.optimal_list[mod][0][4][grp_ind]
                    # )
                X_test_n[mod] = X_test_n_curr
        else:
            X_test_n = [X.copy()]
        self.X_test = X_test_n.copy()
        return X_test_n

    def __pred_common(self, X):
        """
        This function performs the prediction for the DNN learner

        Parameters
        ----------
        X : {array-like, sparse-matrix} of shape (n_test_samples, n_features)
            The input samples.
        """
        if not self.group_stacking:
            X = [X[0].copy() for i in range(self.n_ensemble)]

        n_layer = len(self.optimal_list[0][0][0]) - 1
        for ind_mod, mod in enumerate(self.optimal_list):
            X_test_scaled = X[ind_mod].copy()
            for j in range(n_layer):
                if not self.group_stacking:
                    if len(self.list_continuous) > 0:
                        X_test_scaled[:, self.list_continuous] = mod[1][0].transform(
                            X_test_scaled[:, self.list_continuous]
                        )
                if j == 0:
                    pred = relu(X_test_scaled.dot(mod[0][0][j]) + mod[0][1][j])
                else:
                    pred = relu(pred.dot(mod[0][0][j]) + mod[0][1][j])

            self.pred[ind_mod] = pred.dot(mod[0][0][n_layer]) + mod[0][1][n_layer]

    def set_params(self, **kwargs):
        """
        This function sets the parameters for the DNN estimator
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def clone(self):
        """
        This function clones the DNN learner
        """
        return type(self)(**self.get_params())
