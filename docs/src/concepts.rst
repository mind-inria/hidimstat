.. _concepts:


======================
Definition of concepts
======================

Variable Importance
-------------------

Variable Importance (VI) is a measure of how much a variable contributes to 
the predictive power of a model. We can then define "important" variables
as those whose absence degrades the model's performance
:footcite:p:`Covert2020`.

So if ``VI`` is a variable importance method, ``X`` a variable matrix and ``y`` 
the target variable, the importance of a variable can be estimated as follows:

.. code-block::

    # instantiate the object
    vi = VI()
    # fit the models in the method
    vi.fit(X, y)
    # compute the importance and the pvalues
    importance = vi.importance(X, y)
    # get importance for each feature
    importance = vi.importances_
    # get pvalues
    pvalue = vi.pvalues_                               


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

References
----------

.. footbibliography::
