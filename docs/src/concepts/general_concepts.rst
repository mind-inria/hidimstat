.. _general_concepts:


======================
Definition of concepts
======================

Variable Importance
-------------------

Global Variable Importance (VI) aims to assign a measure of
relevance to each feature :math:`X^j` with respect to a target  :math:`Y` in the
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

Here, ``VI`` can be a variable importance method that inherits from :class:`hidimstat.base_variable_importance.BaseVariableImportance`

Variable Selection
-------------------

(Controlled) Variable selection is then the next step that entails filtering out the
significant features in a way that provides statistical guarantees,
e.g. type-I error or False Discovery Rate (FDR).

For example, if we want to select the variables with a p-value lower than
a threshold ``p``, we can do:

.. code-block::

    # selection of the importance and pvalues
    vi.pvalue_selection(threshold_max=p)

Similarly, we could use ``VI.fdr_selection`` or ``VI.fwer_selection`` to obtain
FDR or FWER control.

This step is important to make insighful discoveries. Even if variable
importance provides a ranking, due to the estimation step, we need
statistical control to do reliable selection.

Variable Selection vs Variable Importance
------------------------------------------

In the literature, there is a gap between *variable selection* and
*variable importance*, as most methods are dedicated to one of these goals
exclusively :footcite:p:`reyerolobo2025principledapproachcomparingvariable`.
For instance, Conditional Feature Importance (:class:`hidimstat.CFI`) typically
serves only as a measure of importance without offering statistical guarantees,
whereas Model-X Knockoffs (:class:`hidimstat.ModelXKnockoff`) generally
provide selection but little beyond that. For this reason, we have adapted the
methods to provide both types of information while preserving their standard
names.



Types of VI methods
-------------------

There are two main types of VI methods implemented in HiDimStat:

1. **Marginal methods**: these methods provide importance to all the features
are related with testing if :math:`X^j\perp\!\!\!\!\perp Y`.
An example of such methods is Leave One Covariate In (LOCI,
:footcite:p:`ewald_2024`).

2. **Conditional methods**: these methods assign importance only to features that
i.e., they contribute unique knowledge. They are related to Conditional
Independence Testing, which consists of testing whether
:math:`X^j\perp\!\!\!\!\perp Y\mid X^{-j}`. Examples of such methods are
:class:`hidimstat.LOCO` and :class:`hidimstat.CFI`.


Generally, conditional methods address the issue of false positives that often
arise with marginal methods, which may assign importance to variables just
because they are correlated with truly important ones. By focusing on unique
contributions, conditional methods help preserve parsimony, yielding a smaller
and more meaningful subset of important features. However, in certain cases, the
distinction between marginal and conditional methods can be more subtle. See
:ref:`sphx_glr_generated_gallery_examples_plot_conditional_vs_marginal_xor_data.py`


High-dimension and correlation
-------------------------------

In high-dimensional and highly correlated settings, estimation becomes
particularly challenging, as it is difficult to clearly distinguish important
features from unimportant ones. For such problems, a preliminary filtering step
can be applied to avoid having duplicate or redundant input features, or
alternatively, one can consider grouping them :footcite:p:`Chamma_AAAI2024` .
Grouping consists of treating together features that represent the same
underlying concept. This approach extends naturally to many methods,
for example :class:`hidimstat.CFI`.



Statistical Inference
---------------------

Given the variability inherent in estimation, it is necessary to apply
statistical control to the discoveries made. Simply selecting the most important
features without such control is not valid. Different forms of guarantees can
be employed, such as controlling the type-I error or the False Discovery Rate.
This step is directly related to the task of Variable Selection.






References
----------

.. footbibliography::
