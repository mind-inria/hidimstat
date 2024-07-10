import numpy as np
from sklearn.base import BaseEstimator

from .Dnn_learner_single import DNN_learner_single
from .utils import sigmoid, softmax


class DNN_learner(BaseEstimator):
    """ToDo
    Parameters
    ----------
    encode: bool, default=False
        Encoding the categorical outcome.
    do_hypertuning: bool, default=True
        Tuning the hyperparameters of the provided estimator.
    dict_hypertuning: dict, default=None
        The dictionary of hyperparameters to tune.
    n_ensemble: int, default=10
        The number of sub-DNN models to fit to the data.
    min_keep: int, default=10
        The minimal number of DNNs to be kept.
    batch_size: int, default=32
        The number of samples per batch for training.
    batch_size_val: int, default=128
        The number of samples per batch for validation.
    n_epoch: int, default=200
        The number of epochs for the DNN learner(s).
    verbose: int, default=0
        If verbose > 0, the fitted iterations will be printed.
    bootstrap: bool, default=True
        Application of bootstrap sampling for the training set.
    split_perc: float, default=0.8
        The training/validation cut for the provided data.
    problem_type: str, default='regression'
        A classification or a regression problem.
    list_cont: list, default=None
        The list of continuous variables.
    list_grps: list of lists, default=None
        A list collecting the indices of the groups' variables
        while applying the stacking method.
    beta1: float, default=0.9
        The exponential decay rate for the first moment estimates.
    beta2: float, default=0.999
        The exponential decay rate for the second moment estimates.
    lr: float, default=1e-3
        The learning rate.
    epsilon: float, default=1e-8
        A small constant added to the denominator to prevent division by zero.
    l1_weight: float, default=1e-2
        The L1-regularization paramter for weight decay.
    l2_weight: float, default=0
        The L2-regularization paramter for weight decay.
    n_jobs: int, default=1
        The number of workers for parallel processing.
    group_stacking: bool, default=False
        Apply the stacking-based method for the provided groups.
    inp_dim: list, default=None
        The cumsum of inputs after the linear sub-layers.
    random_state: int, default=2023
        Fixing the seeds of the random generator.
    Attributes
    ----------
    ToDO
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
        bootstrap=True,
        split_perc=0.8,
        problem_type="regression",
        list_cont=None,
        list_grps=None,
        beta1=0.9,
        beta2=0.999,
        lr=1e-2,
        epsilon=1e-8,
        l1_weight=1e-2,
        l2_weight=1e-2,
        n_jobs=1,
        group_stacking=False,
        inp_dim=None,
        random_state=2023,
    ):
        self.list_estimators = []
        self.encode = encode
        self.do_hypertuning = do_hypertuning
        self.dict_hypertuning = dict_hypertuning
        self.n_ensemble = n_ensemble
        self.min_keep = min_keep
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.n_epoch = n_epoch
        self.verbose = verbose
        self.bootstrap = bootstrap
        self.split_perc = split_perc
        self.problem_type = problem_type
        self.list_grps = list_grps
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.epsilon = epsilon
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.list_cont = list_cont
        self.n_jobs = n_jobs
        self.group_stacking = group_stacking
        self.inp_dim = inp_dim
        self.random_state = random_state
        self.pred = [None] * n_ensemble
        self.enc_y = []
        self.link_func = {
            "classification": softmax,
            "ordinal": sigmoid,
            "binary": sigmoid,
        }
        self.is_encoded = False
        self.dim_repeat = 1

    def fit(self, X, y=None):
        """Build the DNN learner with the training set (X, y)
        Parameters
        ----------
        X : {pandas dataframe}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        if (len(X.shape) != 3) or (X.shape[0] != y.shape[-1]):
            X = np.squeeze(X)
            X = np.array([X for i in range(y.shape[-1])])
            self.dim_repeat = y.shape[-1]

        self.list_estimators = [None] * y.shape[-1]
        self.X_test = [None] * y.shape[-1]

        for y_col in range(y.shape[-1]):
            self.list_estimators[y_col] = DNN_learner_single(
                encode=self.encode,
                do_hypertuning=self.do_hypertuning,
                dict_hypertuning=self.dict_hypertuning,
                n_ensemble=self.n_ensemble,
                min_keep=self.min_keep,
                batch_size=self.batch_size,
                batch_size_val=self.batch_size_val,
                n_epoch=self.n_epoch,
                verbose=self.verbose,
                bootstrap=self.bootstrap,
                split_perc=self.split_perc,
                problem_type=self.problem_type,
                list_cont=self.list_cont,
                list_grps=self.list_grps,
                beta1=self.beta1,
                beta2=self.beta2,
                lr=self.lr,
                epsilon=self.epsilon,
                l1_weight=self.l1_weight,
                l2_weight=self.l2_weight,
                n_jobs=self.n_jobs,
                group_stacking=self.group_stacking,
                inp_dim=self.inp_dim,
                random_state=self.random_state,
            )

            self.list_estimators[y_col].fit(X[y_col, ...], y[:, [y_col]])

        return self

    def hyper_tuning(
        self,
        X_train,
        y_train,
        X_valid,
        y_valid,
        list_hyper=None,
        random_state=None,
    ):
        estimator = DNN_learner_single(
            encode=self.encode,
            do_hypertuning=self.do_hypertuning,
            dict_hypertuning=self.dict_hypertuning,
            n_ensemble=self.n_ensemble,
            min_keep=self.min_keep,
            batch_size=self.batch_size,
            batch_size_val=self.batch_size_val,
            n_epoch=self.n_epoch,
            verbose=self.verbose,
            bootstrap=self.bootstrap,
            split_perc=self.split_perc,
            problem_type=self.problem_type,
            list_cont=self.list_cont,
            list_grps=self.list_grps,
            beta1=self.beta1,
            beta2=self.beta2,
            lr=self.lr,
            epsilon=self.epsilon,
            l1_weight=self.l1_weight,
            l2_weight=self.l2_weight,
            n_jobs=self.n_jobs,
            group_stacking=self.group_stacking,
            inp_dim=self.inp_dim,
            random_state=self.random_state,
        )
        return estimator.hyper_tuning(
            X_train, y_train, X_valid, y_valid, list_hyper, random_state
        )

    def predict(self, X, scale=True):
        """Predict regression target for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        scale: bool, default=True
            The continuous features will be standard scaled or not.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        if isinstance(X, list):
            X = [self.check_X_dim(el) for el in X]
        else:
            X = self.check_X_dim(X)
        list_res = []
        for estimator_ind, estimator in enumerate(self.list_estimators):
            if isinstance(X, list):
                curr_X = [el[estimator_ind, ...] for el in X]
                list_res.append(estimator.predict(curr_X, scale))
            else:
                list_res.append(estimator.predict(X[estimator_ind, ...], scale))
                self.X_test[estimator_ind] = estimator.X_test.copy()
        return np.array(list_res)

    def predict_proba(self, X, scale=True):
        """Predict class probabilities for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        scale: bool, default=True
            The continuous features will be standard scaled or not.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        if isinstance(X, list):
            X = [self.check_X_dim(el) for el in X]
        else:
            X = self.check_X_dim(X)

        list_res = []
        for estimator_ind, estimator in enumerate(self.list_estimators):
            if isinstance(X, list):
                curr_X = [el[estimator_ind, ...] for el in X]
                list_res.append(estimator.predict_proba(curr_X, scale))
            else:
                list_res.append(estimator.predict_proba(X[estimator_ind, ...], scale))
                self.X_test[estimator_ind] = estimator.X_test.copy()
        return np.squeeze(np.array(list_res))

    def set_params(self, **kwargs):
        """Set the parameters of this estimator."""
        for key, value in kwargs.items():
            setattr(self, key, value)
            for estimator in self.list_estimators:
                setattr(estimator, key, value)

    def check_X_dim(self, X):
        if (len(X.shape) != 3) or (X.shape[0] != self.dim_repeat):
            X = np.squeeze(X)
            X = np.array([X for i in range(self.dim_repeat)])

        return X

    def encode_outcome(self, y, train=True):
        for y_col in range(y.shape[-1]):
            y_enc = self.list_estimators[y_col].encode_outcome(y, train=train)
        return y_enc
