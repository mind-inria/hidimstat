.. _slm_methods:


Sparse Linear Models
********************

The methods presented in this section assume that the dependencies between variables can be captured by a linear model. 
They are specifically designed to tackle feature importance and selection in 
high-dimensional settings by leveraging sparse linear models (SLM).  

.. contents:: Table of Contents
   :local:
   :depth: 2


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


Model-X Knockoffs
=================

The Model-X Knockoffs method :footcite:t:`candes2018panning` is a method for variable
selection that controls the false discovery rate (FDR). It is particularly useful in 
high-dimensional settings since its default implementation leverages sparse regression
with the Lasso.  


.. figure:: ./generated/gallery/examples/images/sphx_glr_plot_knockoffs_wisconsin_001.png
    :target: ./generated/gallery/examples/plot_knockoffs_wisconsin.html
    :align: center
    

Regression example
------------------
Model-X Knockoffs can be used as follows:: 

    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import LassoCV
    >>> from hidimstat import ModelXKnockoff

    
    >>> X, y = make_regression(n_features=2)
    
    >>> ko = ModelXKnockoff(estimator=LassoCV(), random_state=0)
    >>> ko.fit(X, y)

    >>> # Selection based on FDR control
    >>> selected_features = ko.fdr_selection(fdr=0.05)


Target quantity
---------------

Model-X Knockoffs is a variable selection method that is not meant to estimate any 
particular measure of importance. The `ModelXKnockoff.importance()` method still returns
the so-called Knockoff statistics, such as the Lasso coefficient difference statistics, 
which can be used to rank the selected variables. The main goal of the method is to 
select a set of features denoted :math:`\hat{S} \subset \{1, \ldots, p\}` such that the 
:term:`false discovery rate (FDR) <FDR>` is controlled at a target 
level. 


Estimation procedure
--------------------

The implementation of :class:`hidimstat.ModelXKnockoff` relies on two main ingredients. 
First, a procedure to generate knockoff variables, that we denote :math:`\tilde{X}`. 
Second, a knockoff statistic. 

Knockoff variables construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given the original design matrix :math:`X = [X^1, \ldots, X^p]`, knockoff variables
are a new set of variables :math:`\tilde{X} = [\tilde{X}^1, \ldots, \tilde{X}^p]` that
have two key properties: 1. they are exchangeable with the original variables. That is,
swapping any subset of variables with their knockoff counterparts does not change the
joint distribution. For example, for :math:`p=3`, 

.. math::
    (X^1, X^2, X^3, \tilde{X}^1, \tilde{X}^2, \tilde{X}^3) \overset{d}{=}
    (\tilde{X}^1, \tilde{X}^2, X^3, X^1, X^2, \tilde{X}^3)

where :math:`\overset{d}{=}` denotes equality in distribution, where the subset of swapped
variables is :math:`\{1, 2\}`. 

In :class:`hidimstat.ModelXKnockoff`, the knockoff variables are generated using the 
equicorrelated construction proposed by :footcite:t:`candes2018panning`. 

Knockoff statistics
^^^^^^^^^^^^^^^^^^^

To perform variable selection, a test statistic needs to be used in order to 
provide evidence against the null hypothesis, which for the :math:`j^{th}` feature is 
:math:`X^j \perp\!\!\!\!\perp Y | X^{-j}`. This statistic needs to satisfy the so-called
flip-sign property, which ensures that swapping a variable with its knockoff counterpart
results in a sign change of the statistic. :class:`hidimstat.ModelXKnockoff` implements 
the popular Lasso Coeffcient-Difference (LCD), which given a Lasso model fitted to 
predict the target :math:`Y` using both the original and knockoff variables 
:math:`[X, \tilde{X}]`, is defined as:

.. math::
    w_j = |\hat{\beta}_j| - |\hat{\beta}_{j + p}| 

where :math:`\hat{\beta}_j` is the Lasso coefficient associated to the original variable
:math:`X^j`, and :math:`\hat{\beta}_{j + p}` is the coefficient associated to its
knockoff counterpart :math:`\tilde{X}^j`. intuitively, a large positive value of 
:math:`w_j` indicates that the original variable contains information that is not 
captured by its knockoff counterpart, and thus provides additional information about the
response :math:`Y`, that is not explained by other variables.


Inference 
---------

The variable selection set :math:`\hat{S}` is obtained by choosing a threshold :math:`\tau` on
the knockoff statistics :math:`w_j`, such that all variables with a statistic larger than
:math:`\tau` are selected. For a target FDR level :math:`\alpha`, choosing the threshold
as:

.. math::
    \tau = \min \left\{ t > 0 : \frac{1 + \#\{j : w_j \leq -t\}}{\#\{j : w_j \geq t\}} \leq \alpha \right\}

guarantees that the FDR is controlled at level :math:`\alpha`.


De-randomization of Knockoffs
-----------------------------

The generation of knockoff variables introduces randomness in the selection procedure. 
Indeed, the sample of knockoff variables :math:`\tilde{X}` corresponds to a single draw 
and repeating the entire procedure, with a different set of knockoff variables, may lead 
to a different selection set. To mitigate this source of variability, the selection can 
be de-randomized by aggregating the results of multiple runs of the Knockoff procedure. 
This can be done using the ``n_repeats`` parameter, for instance,
``ModelXKnockoff(n_repeats=10)``. 


Examples
--------

.. minigallery:: hidimstat.ModelXKnockoff


References
----------
.. footbibliography::


Distilled Conditional Randomization Test 
========================================

Examples
--------
.. minigallery:: hidimstat.D0CRT