import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from tqdm import tqdm

from hidimstat._utils.utils import check_random_state
from hidimstat.base_variable_importance import BaseVariableImportance
from hidimstat.samplers.utils import _subsampling
from hidimstat.statistical_tools.aggregation import quantile_aggregation


class EnsembleImportance(BaseVariableImportance):
    """
    Ensemble inference with arbitrary variable importance measure. Performs multiple runs
    of clustered inference using different clustering obtained from random subsamples
    of the data. The results are then aggregated to provide robust feature importance
    scores and p-values. This algorithm is based on the method described in
    :footcite:`chevalier2022spatially`.

    Parameters
    ----------
    vim: hidimstat.BaseVariableImportance
        Any variable importance method that derives from hidimstat's BaseVariableImportance class.
    n_repeats: int, optional (default=25)
        Number of bootstrap iterations for ensemble inference.
    bootstrap_frac: float, optional (default=0.3)
        Fraction of samples used for the ensemble.
        When bootstrap_frac=1.0, all samples are used.
    bootstrap_groups: ndarray, shape (n_samples,), optional (default=None)
        Sample group labels for stratified subsampling.
    n_jobs : int or None, optional (default=1)
        Number of parallel jobs.
    random_state: int, optional (default=None)
        Random seed for reproducible subsampling.
    ensembling_method : str, optional (default='quantiles')
        Method used for ensembling. Currently, the two available methods
        are 'quantiles' and 'median'.
    gamma : float, optional (default=0.2)
        Lowest gamma-quantile considered to compute the adaptive
        quantile aggregation formula. This parameter is used only if
        `ensembling_method` is 'quantiles'.
    adaptive_aggregation : bool, optional (default=True)
        Whether to use adaptive quantile aggregation when `ensembling_method`
        is 'quantiles'.

    Attributes
    ----------
    ensemble_vims_ : list of hidimstat.BaseVariableImportance
        List of fitted variable importance methods from each bootstrap.
    importances_ : ndarray, shape (n_features,) or (n_features, n_tasks)
        Estimated coefficients at feature level.
    pvalues_ : ndarray, shape (n_features,)
        P-values for each feature.

    .. footbibliography::
    """

    def __init__(
        self,
        vim,
        n_repeats=25,
        bootstrap_frac=0.3,
        bootstrap_groups=None,
        n_jobs=1,
        random_state=None,
        ensembling_method="quantiles",
        gamma=0.5,
        adaptive_aggregation=False,
    ):
        self.vim = vim
        self.n_repeats = n_repeats
        self.bootstrap_frac = bootstrap_frac
        self.bootstrap_groups = bootstrap_groups
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.ensembling_method = ensembling_method
        self.gamma = gamma
        self.adaptive_aggregation = adaptive_aggregation

        self.vim_ = None
        self.ensemble_vims_ = None

    @staticmethod
    def _joblib_fit_one(
        vim,
        bootstrap_frac,
        bootstrap_groups,
        X,
        y,
        random_state,
    ):
        ensemble_samples = _subsampling(
            n_samples=X.shape[0],
            train_size=bootstrap_frac,
            groups=bootstrap_groups,
            random_state=random_state,
        )

        vim_ = clone(vim)
        return vim_.fit(X[ensemble_samples, :], y[ensemble_samples])

    def fit(self, X, y):
        """
        Fit multiple clustered inferences on random subsamples of the data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data matrix.
        y : ndarray, shape (n_samples,) or (n_samples, n_tasks)
            Target variable(s).

        Returns
        -------
        self : EnsembleImportance
            Fitted estimator.
        """
        rng = check_random_state(self.random_state)

        self.ensemble_vims_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_fit_one)(
                vim=self.vim,
                bootstrap_frac=self.bootstrap_frac,
                bootstrap_groups=self.bootstrap_groups,
                X=X,
                y=y,
                random_state=rng_spawned,
            )
            for rng_spawned in tqdm(
                rng.spawn(self.n_repeats),
                desc="Fitting ensemble inferences",
                total=self.n_repeats,
            )
        )

        return self

    @staticmethod
    def _joblib_compute_importance(vim, X, y):
        vim.importance(X, y)
        return vim

    def importance(self, X, y):
        """
        Compute feature importance by aggregating results from multiple
        clustered inferences.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data matrix.
        y : ndarray, shape (n_samples,) or (n_samples, n_tasks)
            Target variable(s).

        Returns
        -------
        importances_ : ndarray, shape (n_features,) or (n_features, n_tasks)
            Estimated importance values at feature level.
        """
        self._check_fit()

        self.ensemble_vims_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._joblib_compute_importance)(
                vim=self.ensemble_vims_[i],
                X=X,
                y=y,
            )
            for i in tqdm(
                range(self.n_repeats),
                desc="Fitting clustered inferences",
                total=self.n_repeats,
            )
        )

        self.importances_ = np.mean(
            [clu_vi.importances_ for clu_vi in self.ensemble_vims_],
            axis=0,
        )

        self.pvalues_ = quantile_aggregation(
            np.array([clu_vi.pvalues_ for clu_vi in self.ensemble_vims_]),
            gamma=self.gamma,
            adaptive=self.adaptive_aggregation,
        )

        return self.importances_

    def fit_importance(self, X, y):
        """
        Fit the model and compute feature importance.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data matrix.
        y : ndarray, shape (n_samples,) or (n_samples, n_tasks)
            Target variable(s).

        Returns
        -------
        importances_ : ndarray, shape (n_features,) or (n_features, n_tasks)
            Estimated importance values at feature level.
        """
        self.fit(X, y)
        self.importance(X, y)
        return self.importances_

    def fdr_selection(
        self,
        fdr,
        fdr_control="bhq",
        reshaping_function=None,
        two_tailed_test=True,
    ):
        """
        Overrides the signature to set two_tailed_test=True by default.
        """
        return super().fdr_selection(
            fdr=fdr,
            fdr_control=fdr_control,
            reshaping_function=reshaping_function,
            two_tailed_test=two_tailed_test,
        )

    def _check_fit(self):
        """
        Check that the ensemble has been fitted.
        """
        if self.ensemble_vims_ is None:
            raise ValueError(
                "The estimators need to be fit before using them."
            )
