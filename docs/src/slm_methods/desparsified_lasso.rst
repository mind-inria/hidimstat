Desparsified Lasso
======================

Desparsified Lasso, also known as debiased Lasso, is a method that aims at estimating 
the regression coefficients :footcite:t:`hastie2015statistical`. To do so, it uses coefficients obtained from a Lasso 
regression and corrects the bias induced by the L1-penalty. This method is particularly
useful in high-dimensional settings where the number of features exceeds the number of 
samples.


Regression example
------------------
Desparsified Lasso can be used as follows::

    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import LassoCV
    >>> from hidimstat import DesparsifiedLasso


    >>> X, y = make_regression(n_features=2)
    
    >>> dl = DesparsifiedLasso(estimator=LassoCV(), n_jobs=n_jobs, random_state=0)
    >>> dl.fit(X, y)
    >>> features_importance = dl.importance(X, y)

    >>> # Selection based on FDR control
    >>> selected_features = dl.fdr_selection(fdr=0.05)
    >>> # Selection based on FWER control
    >>> selected_features = dl.fwer_selection(fwer=0.05)


Target quantity
---------------

This method assumes a linear model of the form: :math:`Y = X \beta^\star + \epsilon`, where
:math:`Y` is the response variable, :math:`X` is the design matrix, :math:`\beta` is
the vector of regression coefficients, and :math:`\epsilon` is an additive noise term.
The target quantity estimated by Desparsified Lasso is the vector of regression
coefficients :math:`\beta^\star`. Denoting :math:`\hat{\beta}_\lambda` the Lasso estimator
with regularization parameter :math:`\lambda`, and :math:`\Theta` an approximate inverse 
of the empirical covariance matrix :math:`\hat{\Sigma} = \frac{1}{n} X^T X`, the 
Desparsified Lasso estimator is given by:

.. math::

    \hat{\beta}_{DL} = \hat{\beta}_\lambda + \frac{1}{n} \Theta X^T (Y - X \hat{\beta}_\lambda)


Estimation procedure
--------------------

The provided implementation :class:`hidimstat.DesparsifiedLasso` is based on the one 
proposed by :footcite:t:`zhang2014confidence`. It first fits a Lasso regression to obtain the initial
estimator :math:`\hat{\beta}_\lambda`, and then estimates nodewise Lasso regressions, 
each predicting a feature :math:`X^j` using all other features :math:`X^{-j}`.


Inference 
----------

As described in :footcite:t:`chevalier2021decoding`, under some sparsity assumptions, 
it can be obtained that the Desparsified Lasso estimator is asymptotically normal. This 
property allows to derive confidence intervals and p-values for the regression
coefficients. 


Extensions to spatially structured data
---------------------------------------

When the features have a known spatial structure, :class:`hidimstat.DesparsifiedLasso`
can be suboptimal, identifying scattered elements of the support and making false 
discoveries far from the support. Methods like :class:`hidimstat.CluDL` and 
:class:`hidimstat.EnCluDL` address this issue by leveraging the spatial structure of the
data. Read more in the :ref:`User Guide <high_dimension>`. 


Examples
--------

.. minigallery:: hidimstat.DesparsifiedLasso


References
----------
.. footbibliography::