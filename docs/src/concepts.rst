.. _concepts:


======================
Definition of concepts
======================

Variable Importance
-------------------

Variable Importance (VI) aims to assign a measure of
relevance to each feature :math:`X^j` with respect to a target  :math:`y` in the
data-generating process. In Machine Learning, it can be seen as a measure
of how much a variable contributes to the predictive power of a model. We 
can then define "important" variables as those whose absence degrades 
the model's performance :footcite:p:`Covert2020`.

So if ``VI`` is a variable importance method, ``X`` a variable matrix and ``y`` 
the target variable, the importance of all the variables
can be estimated as follows:

.. code-block::

    # instantiate the object
    vi = VI()
    # fit the models in the method
    vi.fit(X, y)
    # compute the importance and the pvalues
    importance = vi.importance(X, y)
    # get importance for each feature
    importance = vi.importances_

It allow us to rank the variables from more to less important.                            

Here, ``VI`` can be a variable importance method implemented in HiDimStat,
such as :class:`hidimstat.D0CRT` (other methods will support the same API 
soon).

(Controlled) Variable Selection
-------------------------------

Variable selection is then the next step that entails filtering out the 
significant features in a way that provides statistical guarantees, 
e.g. type-I error or False Discovery Rate (FDR).

So, if we want to select the variables with a p-value lower than a threshold 
``p``, we can do:

.. code-block::

    # selection of the importance and pvalues
    vi.selection(threshold_pvalue=p)

Types of VI methods
-------------------

There are two main types of VI methods implemented in HiDimStat:

1. Conditional methods: these methods estimate the importance of a variable
   conditionally to all the other variables. Examples of such methods are
   :class:`hidimstat.LOCO` and :class:`hidimstat.CFI`.

2. Marginal methods: these methods estimate the importance of a variable
   marginally to all the other variables. Examples of such methods are
   :class:`hidimstat.PFI` and :class:`hidimstat.D0CRT`.

The main difference between these two types of methods is that conditional
methods are more computationally expensive but they can handle correlated
variables better than marginal methods :footcite:p:`Covert2020`.

In particular, marginal methods can be too conservative when variables are
highly correlated, leading to a loss of power in the variable selection step.
However, marginal methods are more scalable to high-dimensional datasets
and they can be used when the number of samples is smaller than the number of
variables, which is not the case for conditional methods.


High-dimensionality and correlation
-----------------------------------

Problem: with high-dimension 

Solution: prior filtering of redundant variables or considering grouping. Brief definition of grouping.  



Statistical Inference
---------------------



References
----------

.. footbibliography::
