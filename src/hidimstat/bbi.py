import itertools
import warnings
from copy import copy

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .compute_importance import joblib_compute_conditional, joblib_compute_permutation
from .Dnn_learner import DnnLearner
from .utils import compute_imp_std, convert_predict_proba, create_X_y


class BlockBasedImportance(BaseEstimator, TransformerMixin):
    """
    This class implements the Block-Based Importance (BBI), a framework for
    variable importance computation with statistical guarantees.
    It consists of two blocks of estimators: Learner block (Predicting on the
    data) and Importance block (Resampling the variable/group of interest to
    assess the impact on the loss). For single-level see
    :footcite:t:`Chamma_NeurIPS2023` and for group-level see
    :footcite:t:`Chamma_AAAI2024`.

    Parameters
    ----------
    estimator : {String or sklearn.base.BaseEstimator}, default="DNN"
        The provided estimator for the learner block.
        The default estimator is a custom Multi-Layer Perceptron (MLP) learner.

        - String options include:
            - "DNN" for the Multi-Layer Perceptron
            - "RF" for the Random Forest
        - Other options include:
            - sklearn.base.BaseEstimator
    importance_estimator : {String or sklearn.base.BaseEstimator}, default="sampling_RF"
        The provided estimator for the importance block.
        The default estimator includes the use of the sampling Random Forest
        where the sampling is executed in the corresponding leaf of each
        instance within its neighbors

        - String options include:
            - "sampling_RF" for the sampling Random Forest
            - "residuals_RF" for the Random Forest along with the residuals path
              for importance computation
        - Other options include:
            - sklearn.base.BaseEstimator
    coffeine_transformer : tuple, default=None
        Applying the coffeine's pipeline for filterbank models on
        electrophysiological data.
        The tuple consists of (coffeine pipeline, new number of variables) or
        (coffeine pipeline, new number of variables, list of variables to keep
        after variable selection).
    do_hypertuning : bool, default=True
        Tuning the hyperparameters of the provided estimator.
    dict_hypertuning : dict, default=None
        The dictionary of hyperparameters to tune, depending on the provided
        inference estimator.
    problem_type : str, default='regression'
        A classification or a regression problem.
    encoding_input : bool, default=True
        To one-hot or ordinal encode the nominal and ordinal input variables.
    sampling_with_repetition : bool, default=True
        Sampling with repetition the train part of the train/valid scheme under
        the training set. The number of training samples in train is equal to
        the number of instances in the training set.
    split_percentage : float, default=0.8
        The training/validation cut for the provided data.
    conditional : bool, default=True
        The permutation or the conditional sampling approach.
    variables_categories : dict, default=None
        The dictionary of binary, nominal and ordinal variables.
    residuals_sampling : bool, default=False
        The use of permutations or random sampling for residuals with the
        conditional sampling.
    n_permutations : int, default=50
        The number of permutations/random samplings for each column.
    n_jobs : int, default=1
        The number of workers for parallel processing.
    verbose : int, default=0
        If verbose > 0, the fitted iterations will be printed.
    groups : dict, default=None
        The knowledge-driven/data-driven grouping of the variables if provided.
    group_stacking : bool, default=False
        Apply the stacking-based method for the provided groups.
    sub_groups : dict, default=None
        The list of provided variables's indices to condition on per
        variable/group of interest (default set to all the remaining variables).
    k_fold : int, default=2
        The number of folds for k-fold cross fitting.
    prop_out_subLayers : int, default=0.
        If group_stacking is True, the proportion of outputs for
        the linear sub-layers per group.
    iteration_index : int, default=None
        The index of the current processed iteration.
    random_state : int, default=2023
        Fixing the seeds of the random generator.
    do_compute_importance : boolean, default=True
        Whether to compute the Importance Scores.
    group_fold : list, default=None
        The list of group labels to perform GroupKFold to keep subjects within
        the same training or test set.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        estimator="DNN",
        importance_estimator="sampling_RF",
        coffeine_transformer=None,
        do_hypertuning=True,
        dict_hypertuning=None,
        problem_type="regression",
        encoding_input=True,
        sampling_with_repetition=True,
        split_percentage=0.8,
        conditional=True,
        variables_categories=None,
        residuals_sampling=False,
        n_permutations=50,
        n_jobs=1,
        verbose=0,
        groups=None,
        group_stacking=False,
        sub_groups=None,
        k_fold=2,
        prop_out_subLayers=0,
        iteration_index=None,
        random_state=2023,
        do_compute_importance=True,
        group_fold=None,
    ):
        self.estimator = estimator
        self.importance_estimator = importance_estimator
        self.coffeine_transformer = coffeine_transformer
        self.do_hypertuning = do_hypertuning
        self.dict_hypertuning = dict_hypertuning
        self.problem_type = problem_type
        self.encoding_input = encoding_input
        self.sampling_with_repetition = sampling_with_repetition
        self.split_percentage = split_percentage
        self.conditional = conditional
        self.variables_categories = variables_categories
        self.residuals_sampling = residuals_sampling
        self.n_permutations = n_permutations
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.groups = groups
        self.sub_groups = sub_groups
        self.group_stacking = group_stacking
        self.k_fold = k_fold
        self.prop_out_subLayers = prop_out_subLayers
        self.iteration_index = iteration_index
        self.random_state = random_state
        self.X_test = [None] * max(self.k_fold, 1)
        self.y_test = [None] * max(self.k_fold, 1)
        self.y_train = [None] * max(self.k_fold, 1)
        self.org_pred = [None] * max(self.k_fold, 1)
        self.pred_scores = [None] * max(self.k_fold, 1)
        self.X_nominal = [None] * max(self.k_fold, 1)
        self.type = None
        self.list_estimators = [None] * max(self.k_fold, 1)
        self.X_proc = [None] * max(self.k_fold, 1)
        self.scaler_x = [None] * max(self.k_fold, 1)
        self.scaler_y = [None] * max(self.k_fold, 1)
        self.do_compute_importance = do_compute_importance
        self.group_fold = group_fold
        # Check for applying the stacking approach with the RidgeCV estimator
        self.apply_ridge = False
        # Check for the case of a coffeine transformer with provided groups
        self.transformer_grp = True
        self.coffeine_transformers = []

    def fit(self, X, y=None):
        """
        Build the provided estimator with the training set (X, y)

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
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
        # Disable the importance computation when only one group is provided
        # (group case)
        if self.groups is not None:
            if len(self.groups) <= 1:
                self.do_compute_importance = False

        # Fixing the random generator's seed
        self.rng = np.random.RandomState(self.random_state)

        # Switch to special binary case
        if (self.problem_type == "classification") and (len(np.unique(y)) < 3):
            self.problem_type = "binary"
        # Convert variables_categories to a dictionary if initialized
        # as an empty string
        if not isinstance(self.variables_categories, dict):
            self.variables_categories = {
                "nominal": [],
                "ordinal": [],
                "binary": [],
            }

        if "binary" not in self.variables_categories:
            self.variables_categories["binary"] = []
        if "ordinal" not in self.variables_categories:
            self.variables_categories["ordinal"] = []
        if "nominal" not in self.variables_categories:
            self.variables_categories["nominal"] = []

        # Move the ordinal columns with 2 values to the binary part
        if self.variables_categories["ordinal"] != []:
            for ord_col in set(self.variables_categories["ordinal"]).intersection(
                list(X.columns)
            ):
                if len(np.unique(X[ord_col])) < 3:
                    self.variables_categories["binary"].append(ord_col)
                    self.variables_categories["ordinal"].remove(ord_col)

        # Convert X to pandas dataframe
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        else:
            if self.k_fold == 0:
                X = X.reset_index(drop=True)
            self.X_cols = list(X.columns)
        if (self.groups is None) or (not bool(self.groups)):
            # Initialize the list_columns variable with each feature
            # in a separate list (default case)
            self.groups = [[col] for col in X.columns]
            self.transformer_grp = False
            if self.group_stacking:
                # Remove the group_stacking flag not
                # to complex the DNN architecture
                self.group_stacking = False
                warnings.warn(
                    "The groups are not provided to apply the stacking"
                    " approach, back to single variables case."
                )

        # Convert dictionary of groups to a list of lists
        if isinstance(self.groups, dict):
            self.groups = list(self.groups.values())

        # Checks if the number of variables in the groups is equal to the
        # number of variables provided
        list_count = [item for sublist in self.groups for item in sublist]
        if self.coffeine_transformer is None:
            if len(set(list_count)) != X.shape[1]:
                raise Exception("The provided groups are missing some variables!")
        else:
            if self.transformer_grp:
                if len(set(list_count)) != (X.shape[1] * self.coffeine_transformer[1]):
                    raise Exception("The provided groups are missing some variables!")
            else:
                if len(set(list_count)) != X.shape[1]:
                    raise Exception("The provided groups are missing some variables!")

        # Check if categorical variables exist within the columns of the design
        # matrix
        self.variables_categories["binary"] = list(
            set(self.variables_categories["binary"]).intersection(list(X.columns))
        )
        self.variables_categories["ordinal"] = list(
            set(self.variables_categories["ordinal"]).intersection(list(X.columns))
        )
        self.variables_categories["nominal"] = list(
            set(self.variables_categories["nominal"]).intersection(list(X.columns))
        )

        self.list_columns = self.groups.copy()
        self.list_cat_tot = list(
            itertools.chain.from_iterable(self.variables_categories.values())
        )
        X_nominal_org = X.loc[:, self.list_cat_tot]

        # One-hot encoding of nominal variables
        tmp_list = []
        self.dict_nominal = {}
        # A dictionary to save the encoders of the nominal variables
        self.dict_enc = {}
        if len(self.variables_categories["nominal"]) > 0:
            for col_encode in self.variables_categories["nominal"]:
                enc = OneHotEncoder(handle_unknown="ignore")
                enc.fit(X[[col_encode]])
                labeled_cols = [
                    enc.feature_names_in_[0] + "_" + str(enc.categories_[0][j])
                    for j in range(len(enc.categories_[0]))
                ]
                hot_cols = pd.DataFrame(
                    enc.transform(X[[col_encode]]).toarray(),
                    dtype="int32",
                    columns=labeled_cols,
                )
                X = X.drop(columns=[col_encode])
                X = pd.concat([X, hot_cols], axis=1)
                self.dict_enc[col_encode] = enc

        # Create a dictionary for categorical variables with their indices
        for col_cat in self.list_cat_tot:
            current_list = [
                col
                for col in range(len(X.columns))
                if X.columns[col].split("_")[0] == col_cat
            ]
            if len(current_list) > 0:
                self.dict_nominal[col_cat] = current_list
                # A list to store the labels of the categorical variables
                tmp_list.extend(current_list)

        # Create a dictionary for the continuous variables that will be scaled
        self.dict_continuous = {}
        if self.coffeine_transformer is None:
            for ind_col, col_cont in enumerate(X.columns):
                if ind_col not in tmp_list:
                    self.dict_continuous[col_cont] = [ind_col]
            self.list_continuous = [el[0] for el in self.dict_continuous.values()]
        else:
            self.list_columns_tmp = []
            self.list_continuous = list(
                np.arange(X.shape[1] * self.coffeine_transformer[1])
            )
            for i in range(X.shape[1] * self.coffeine_transformer[1]):
                self.dict_continuous[i] = [i]
                self.list_columns_tmp.append([i])
            if not self.transformer_grp:
                self.list_columns = self.list_columns_tmp.copy()
            self.coffeine_transformers = [
                copy(self.coffeine_transformer[0]) for _ in range(max(self.k_fold, 1))
            ]

        X = X.to_numpy()

        if len(y.shape) != 2:
            y = np.array(y).reshape(-1, 1)

        if self.problem_type in ("classification", "binary"):
            self.loss = log_loss
        else:
            self.loss = mean_squared_error

        # Replace groups' variables by the indices in the design matrix
        self.list_grps = []
        self.input_dimensions = None
        if self.group_stacking:
            for grp in self.groups:
                current_grp = []
                for i in grp:
                    if i in self.dict_nominal.keys():
                        current_grp += self.dict_nominal[i]
                    else:
                        current_grp += self.dict_continuous[i]
                self.list_grps.append(current_grp)

            # To check
            if len(self.coffeine_transformers) == 1:
                X = self.coffeine_transformers[0].fit_transform(
                    pd.DataFrame(X, columns=self.X_cols), np.ravel(y)
                )

            if self.estimator != "DNN":
                # Force the output to 1 neurone per group
                # in standard stacking case
                self.prop_out_subLayers = 0
                # RidgeCV estimators
                self.ridge_folds = 10
                list_alphas = np.logspace(-3, 5, num=100)
                cv = KFold(
                    n_splits=self.ridge_folds,
                    random_state=self.random_state,
                    shuffle=True,
                )
                self.ridge_mods = [
                    [
                        make_pipeline(StandardScaler(), RidgeCV(list_alphas))
                        for __ in range(len(self.list_grps))
                    ].copy()
                    for _ in range(self.ridge_folds)
                ]

                X_prev = X.copy()
                X = np.zeros((y.shape[0], len(self.list_grps)))
                for index_fold, (train, test) in enumerate(cv.split(X_prev)):
                    X_train, X_test = X_prev[train], X_prev[test]
                    y_train, _ = y[train], y[test]
                    if len(self.coffeine_transformers) > 1:
                        X_train = self.coffeine_transformers[index_fold].fit_transform(
                            pd.DataFrame(X_train, columns=self.X_cols),
                            np.ravel(y_train),
                        )
                        X_test = self.coffeine_transformers[index_fold].transform(
                            pd.DataFrame(X_test, columns=self.X_cols)
                        )
                    for grp_ind, grp in enumerate(self.list_grps):
                        self.ridge_mods[index_fold][grp_ind].fit(
                            X_train[:, grp], y_train
                        )
                        X[test, grp_ind] = (
                            self.ridge_mods[index_fold][grp_ind]
                            .predict(X_test[:, grp])
                            .ravel()
                        )
                self.apply_ridge = True

            self.input_dimensions = [
                max(1, int(self.prop_out_subLayers * len(grp)))
                for grp in self.list_grps
            ]
            self.input_dimensions.insert(0, 0)
            self.input_dimensions = np.cumsum(self.input_dimensions)
            self.list_columns = [
                list(
                    np.arange(
                        self.input_dimensions[grp_ind],
                        self.input_dimensions[grp_ind + 1],
                    )
                )
                for grp_ind in range(len(self.list_grps))
            ]

        # Initialize the first estimator (block learner)
        if self.estimator == "DNN":
            self.estimator = DnnLearner(
                encoding_outcome=True,
                problem_type=self.problem_type,
                do_hypertuning=False,
                list_continuous=self.list_continuous,
                list_grps=self.list_grps,
                group_stacking=self.group_stacking,
                n_jobs=self.n_jobs,
                input_dimensions=self.input_dimensions,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            self.type = "DNN"
            # Initializing the dictionary for tuning the hyperparameters
            if self.dict_hypertuning is None:
                self.dict_hypertuning = {
                    "lr": [1e-4, 1e-3, 1e-2],
                    "l1_weight": [0, 1e-4, 1e-2],
                    "l2_weight": [0, 1e-4, 1e-2],
                }

        elif self.estimator == "RF":
            if self.problem_type == "regression":
                self.estimator = RandomForestRegressor(random_state=2023)
            else:
                self.estimator = RandomForestClassifier(random_state=2023)
            self.dict_hypertuning = {"max_depth": [2, 5, 10, 20]}
            self.type = "RF"

        if self.k_fold != 0:
            # Implementing k-fold cross validation as the default behavior
            if self.group_fold:
                kf = GroupKFold(n_splits=self.k_fold)
                list_splits = kf.split(X, y, self.group_fold)
            else:
                kf = KFold(
                    n_splits=self.k_fold,
                    random_state=self.random_state,
                    shuffle=True,
                )
                list_splits = kf.split(X)

            for index_fold, (train_index, test_index) in enumerate(list_splits):
                print(f"Processing: {index_fold+1}")
                X_fold = X.copy()
                y_fold = y.copy()

                self.X_nominal[index_fold] = X_nominal_org.iloc[test_index, :]

                X_train, X_test = (
                    X_fold[train_index, :],
                    X_fold[test_index, :],
                )

                y_train, y_test = y_fold[train_index], y_fold[test_index]

                if not self.apply_ridge:
                    if self.coffeine_transformer is not None:
                        X_train = self.coffeine_transformers[index_fold].fit_transform(
                            pd.DataFrame(X_train, columns=self.X_cols),
                            np.ravel(y_train),
                        )

                        X_test = self.coffeine_transformers[index_fold].transform(
                            pd.DataFrame(X_test, columns=self.X_cols)
                        )

                self.X_test[index_fold] = X_test.copy()
                self.y_test[index_fold] = y_test.copy()
                self.y_train[index_fold] = y_train.copy()

                # Find the list of optimal sub-models to be used in the
                # following steps (Default estimator)
                if self.do_hypertuning:
                    self.__tuning_hyper(X_train, y_train, index_fold)
                if self.type == "DNN":
                    self.estimator.fit(X_train, y_train)
                self.list_estimators[index_fold] = copy(self.estimator)

        else:
            self.y_train = y.copy()
            if not self.apply_ridge:
                if self.coffeine_transformer is not None:
                    X = self.coffeine_transformers[0].fit_transform(
                        pd.DataFrame(X, columns=self.X_cols), np.ravel(y)
                    )
                    if not self.transformer_grp:
                        # Variables are provided as the third element of the
                        # coffeine transformer parameter
                        if len(self.coffeine_transformer) > 2:
                            X = X[:, self.coffeine_transformer[2]]
                            self.list_continuous = np.arange(
                                len(self.coffeine_transformer[2])
                            )

            # Hyperparameter tuning
            if self.do_hypertuning:
                self.__tuning_hyper(X, y, 0)

            if self.type == "DNN":
                self.estimator.fit(X, y)
            self.list_estimators[0] = copy(self.estimator)

        self.is_fitted = True
        return self

    def __tuning_hyper(self, X, y, index_fold=None):
        """
        Tune the hyperparameters of the provided inference estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_train_samples, n_features)
            The training input samples.
        y : array-like of shape (n_train_samples,) or (n_train_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        index_fold : int, default=None
            The index of the corresponding fold.

        """
        if not ((self.apply_ridge) and (self.group_stacking)):
            (
                X_train_scaled,
                y_train_scaled,
                X_validation_scaled,
                y_validation_scaled,
                X_scaled,
                __,
                scaler_x,
                scaler_y,
                ___,
            ) = create_X_y(
                X,
                y,
                sampling_with_repetition=self.sampling_with_repetition,
                split_percentage=self.split_percentage,
                problem_type=self.problem_type,
                list_continuous=self.list_continuous,
                random_state=self.random_state,
            )
            if self.dict_hypertuning is not None:
                list_hypertuning = list(
                    itertools.product(*list(self.dict_hypertuning.values()))
                )
            list_loss = []
            if self.type == "DNN":
                list_loss = self.estimator.hyper_tuning(
                    X_train_scaled,
                    y_train_scaled,
                    X_validation_scaled,
                    y_validation_scaled,
                    list_hypertuning,
                    random_state=self.random_state,
                )
            else:
                if self.dict_hypertuning is None:
                    self.estimator.fit(X_scaled, y)
                    # If not a DNN learner case, need to save the scalers
                    self.scaler_x[index_fold] = scaler_x
                    self.scaler_y[index_fold] = scaler_y
                    return
                else:
                    for ind_el, el in enumerate(list_hypertuning):
                        curr_params = dict(
                            (k, v)
                            for v, k in zip(el, list(self.dict_hypertuning.keys()))
                        )
                        list_hypertuning[ind_el] = curr_params
                        self.estimator.set_params(**curr_params)
                        if self.problem_type == "regression":
                            y_train_curr = (
                                y_train_scaled * scaler_y.scale_ + scaler_y.mean_
                            )
                            y_validation_curr = (
                                y_validation_scaled * scaler_y.scale_ + scaler_y.mean_
                            )

                            def func(x):
                                return self.estimator.predict(x)

                        else:
                            y_train_curr = y_train_scaled.copy()
                            y_validation_curr = y_validation_scaled.copy()

                            def func(x):
                                return self.estimator.predict_proba(x)

                        self.estimator.fit(X_train_scaled, y_train_curr)

                        if self.problem_type == "classification":
                            list_loss.append(
                                self.loss(
                                    y_validation_curr,
                                    func(X_validation_scaled)[
                                        :, np.unique(y_validation_curr)
                                    ],
                                )
                            )
                        else:
                            list_loss.append(
                                self.loss(y_validation_curr, func(X_validation_scaled))
                            )

            ind_min = np.argmin(list_loss)
            best_hyper = list_hypertuning[ind_min]
            if not isinstance(best_hyper, dict):
                best_hyper = dict(zip(self.dict_hypertuning.keys(), best_hyper))

            self.estimator.set_params(**best_hyper)
            self.estimator.fit(X_scaled, y)

            # If not a DNN learner case, need to save the scalers
            self.scaler_x[index_fold] = scaler_x
            self.scaler_y[index_fold] = scaler_y

        else:
            self.estimator.fit(X, y)

    def predict(self, X=None):
        """
        This function predicts the regression target for the input samples X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_test_samples, n_features),
            default=None
            The input samples.

        Returns
        -------
        predictions: array-like of shape (n_test_samples,)
            The average predictions across all folds.
        """
        if not isinstance(X, list):
            list_X = [X.copy() for el in range(max(self.k_fold, 1))]
            mean_pred = True
        else:
            list_X = X.copy()
            mean_pred = False

        for index_fold, curr_X in enumerate(list_X):
            # Prepare the test set for the prediction
            if self.encoding_input:
                X_tmp = self._encode_input(curr_X)
            else:
                X_tmp = curr_X.copy()

            if self.type != "DNN":
                if not isinstance(curr_X, np.ndarray):
                    X_tmp = np.array(X_tmp)
                if self.scaler_x[index_fold] is not None:
                    X_tmp[:, self.list_continuous] = self.scaler_x[
                        index_fold
                    ].transform(X_tmp[:, self.list_continuous])
                self.X_proc[index_fold] = [X_tmp.copy()]

            self.org_pred[index_fold] = self.list_estimators[index_fold].predict(X_tmp)

            # Convert to the (n_samples x n_outputs) format
            if len(self.org_pred[index_fold].shape) != 2:
                self.org_pred[index_fold] = self.org_pred[index_fold].reshape(-1, 1)

            if self.type == "DNN":
                self.X_proc[index_fold] = np.array(
                    self.list_estimators[index_fold].X_test.copy()
                ).swapaxes(0, 1)

        if mean_pred:
            return np.mean(np.array(self.org_pred), axis=0)

    def predict_proba(self, X=None):
        """
        This function predicts the class probabilities for the input samples X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_test_samples, n_features),
            default=None
            The input samples.

        Returns
        -------
        predictions: array-like of shape (n_test_samples,)
            The average predictions across all folds.
        """
        if not isinstance(X, list):
            list_X = [X.copy() for el in range(max(self.k_fold, 1))]
            mean_pred = True
        else:
            list_X = X.copy()
            mean_pred = False

        for index_fold, curr_X in enumerate(list_X):
            # Prepare the test set for the prediction
            if self.encoding_input:
                X_tmp = self._encode_input(curr_X)
            else:
                X_tmp = curr_X.copy()

            if self.type != "DNN":
                if not isinstance(curr_X, np.ndarray):
                    X_tmp = np.array(X_tmp)
                if self.scaler_x[index_fold] is not None:
                    X_tmp[:, self.list_continuous] = self.scaler_x[
                        index_fold
                    ].transform(X_tmp[:, self.list_continuous])
                self.X_proc[index_fold] = [X_tmp.copy()]

            self.org_pred[index_fold] = self.list_estimators[index_fold].predict_proba(
                X_tmp
            )

            if self.type == "DNN":
                self.X_proc[index_fold] = np.array(
                    self.list_estimators[index_fold].X_test.copy()
                ).swapaxes(0, 1)
            else:
                self.org_pred[index_fold] = convert_predict_proba(
                    self.org_pred[index_fold]
                )

        if mean_pred:
            return np.mean(np.array(self.org_pred), axis=0)

    def _encode_input(self, X):
        """
        This function encodes the non-continuous variables in the design matrix
        X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_test_samples, n_features)
            The input samples.

        Returns
        -------
        X : {array-like, sparse matrix}
            The new encoded design matrix.
        """
        # Check is fit had been called
        check_is_fitted(self, ["is_fitted"])

        # One-hot encoding for the test set
        if len(self.variables_categories["nominal"]) > 0:
            for col_encode in self.variables_categories["nominal"]:
                enc = self.dict_enc[col_encode]
                labeled_cols = [
                    enc.feature_names_in_[0] + "_" + str(enc.categories_[0][j])
                    for j in range(len(enc.categories_[0]))
                ]
                hot_cols = pd.DataFrame(
                    enc.transform(X[[col_encode]]).toarray(),
                    dtype="int32",
                    columns=labeled_cols,
                )
                X = X.drop(columns=[col_encode])
                X = pd.concat([X, hot_cols], axis=1)

        return X

    def compute_importance(self, X=None, y=None):
        """
        This function computes the importance scores and the statistical
        guarantees per variable/group of interest

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_test_samples, n_features)
            The training input samples.
        y : array-like of shape (n_test_samples,) or (n_test_samples, n_outputs),
            default=None
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        results : dict
            The dictionary of importance scores, p-values and the corresponding
            z-scores.
        """
        # Check is fit had been called
        check_is_fitted(self, ["is_fitted"])
        self.encoding_input = True

        if self.k_fold != 0:
            X = self.X_test.copy()
            y = self.y_test.copy()
            self.encoding_input = False
        else:
            if self.coffeine_transformer is not None:
                X = self.coffeine_transformers[0].transform(
                    pd.DataFrame(X, columns=self.X_cols)
                )
                if not self.transformer_grp:
                    # Variables are provided as the third element of the
                    # coffeine transformer parameter
                    if len(self.coffeine_transformer) > 2:
                        X = X[:, self.coffeine_transformer[2]]
                        self.list_continuous = np.arange(
                            len(self.coffeine_transformer[2])
                        )
            # Perform stacking if enabled
            if self.apply_ridge:
                X_prev = X.copy()
                X = np.zeros((y.shape[0], len(self.list_grps), self.ridge_folds))
                for grp_ind, grp in enumerate(self.list_grps):
                    for ind_ridge in range(self.ridge_folds):
                        X[:, grp_ind, ind_ridge] = (
                            self.ridge_mods[ind_ridge][grp_ind]
                            .predict(X_prev.iloc[:, grp])
                            .ravel()
                        )
                X = np.mean(X, axis=-1)

            # Convert X to pandas dataframe
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            else:
                X = X.reset_index(drop=True)
            self.X_nominal[0] = X.loc[:, self.list_cat_tot]
            X = [X.copy() for _ in range(max(self.k_fold, 1))]
            if self.problem_type in ("classification", "binary"):
                pass
            else:
                if len(y.shape) != 2:
                    y = y.reshape(-1, 1)
            y = [y.copy() for _ in range(max(self.k_fold, 1))]

        # Compute original predictions
        if self.problem_type == "regression":
            output_dimension = y[0].shape[1]
            self.predict(X)
        else:
            output_dimension = 1
            self.predict_proba(X)
        list_seeds_imp = self.rng.randint(1e5, size=self.n_permutations)
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        score_imp_l = []
        results = {}
        # n_features x n_permutations x n_samples
        for index_fold, estimator in enumerate(self.list_estimators):
            if self.type == "DNN":
                for y_col in range(y[index_fold].shape[-1]):
                    _ = self.estimator.encode_outcome(
                        self.y_train[index_fold], train=True
                    )[y_col]
                    y[index_fold] = self.estimator.encode_outcome(
                        y[index_fold], train=False
                    )[y_col]
            else:
                if self.problem_type in ("classification", "binary"):
                    one_hot = OneHotEncoder(handle_unknown="ignore").fit(
                        self.y_train[index_fold].reshape(-1, 1)
                    )
                    y[index_fold] = one_hot.transform(
                        y[index_fold].reshape(-1, 1)
                    ).toarray()
            if self.do_compute_importance:
                if not self.conditional:
                    self.pred_scores[index_fold], score_cur = list(
                        zip(
                            *parallel(
                                delayed(joblib_compute_permutation)(
                                    self.list_columns[variables_interest],
                                    permutation,
                                    estimator,
                                    self.type,
                                    self.X_proc[index_fold],
                                    y[index_fold],
                                    self.problem_type,
                                    self.org_pred[index_fold],
                                    dict_continuous=self.dict_continuous,
                                    dict_nominal=self.dict_nominal,
                                    processed_column=variables_interest,
                                    iteration_index=index_fold + 1,
                                    group_stacking=self.group_stacking,
                                    random_state=list_seeds_imp[permutation],
                                    verbose=self.verbose,
                                )
                                for variables_interest in range(len(self.list_columns))
                                for permutation in range(self.n_permutations)
                            )
                        )
                    )
                    self.pred_scores[index_fold] = np.array(
                        self.pred_scores[index_fold]
                    ).reshape(
                        (
                            len(self.list_columns),
                            self.n_permutations,
                            y[index_fold].shape[0],
                            output_dimension,
                        )
                    )
                else:
                    self.pred_scores[index_fold], score_cur = list(
                        zip(
                            *parallel(
                                delayed(joblib_compute_conditional)(
                                    self.list_columns[variables_interest],
                                    self.n_permutations,
                                    estimator,
                                    self.type,
                                    self.importance_estimator,
                                    self.X_proc[index_fold],
                                    y[index_fold],
                                    self.problem_type,
                                    self.org_pred[index_fold],
                                    seed=self.random_state,
                                    dict_continuous=self.dict_continuous,
                                    dict_nominal=self.dict_nominal,
                                    X_nominal=self.X_nominal[index_fold],
                                    variables_categories=self.variables_categories,
                                    encoder=self.dict_enc,
                                    processed_column=variables_interest,
                                    iteration_index=index_fold + 1,
                                    group_stacking=self.group_stacking,
                                    sub_groups=[self.list_columns, self.sub_groups],
                                    list_seeds=list_seeds_imp,
                                    residuals_sampling=self.residuals_sampling,
                                    output_dimension=output_dimension,
                                    verbose=self.verbose,
                                )
                                for variables_interest in range(len(self.list_columns))
                            )
                        )
                    )
                    self.pred_scores[index_fold] = np.array(
                        self.pred_scores[index_fold]
                    )
                score_imp_l.append(score_cur[0])
            else:
                if self.problem_type in ("classification", "binary"):
                    nonzero_cols = np.where(y[index_fold].any(axis=0))[0]
                    score = roc_auc_score(
                        y[index_fold][:, nonzero_cols],
                        self.org_pred[index_fold][:, nonzero_cols],
                    )
                else:
                    score = (
                        mean_absolute_error(y[index_fold], self.org_pred[index_fold]),
                        r2_score(y[index_fold], self.org_pred[index_fold]),
                    )
                score_imp_l.append(score)

        # Compute performance
        if self.problem_type == "regression":
            results["score_MAE"] = np.mean(np.array(score_imp_l), axis=0)[0]
            results["score_R2"] = np.mean(np.array(score_imp_l), axis=0)[1]
        else:
            results["score_AUC"] = np.mean(np.array(score_imp_l), axis=0)

        if not self.do_compute_importance:
            return results

        # Compute Importance and P-values
        pred_scores_full = [
            np.mean(self.pred_scores[index_fold], axis=1)
            for index_fold in range(max(self.k_fold, 1))
        ]
        results["importance"] = compute_imp_std(pred_scores_full)[0]
        results["std"] = compute_imp_std(pred_scores_full)[1]
        results["pval"] = norm.sf(results["importance"] / results["std"])
        results["pval"][np.isnan(results["pval"])] = 1

        # Compute Z-scores across Permutations/Sampling
        imp_z = compute_imp_std(self.pred_scores)[0]
        std_z = compute_imp_std(self.pred_scores)[1]
        results["zscores"] = imp_z / std_z
        return results
