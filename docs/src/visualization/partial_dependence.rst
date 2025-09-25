
.. _partial_dependance_plots:

========================
Partial Dependence plots
========================

Definition
==========

A Partial Dependence Plot (PDP) :footcite:`friedman2001greedy` is a tool used to visualize
the dependence between the target variable and a feature (or set of features) of interest.
For important features, it can be used to understand their relationship with the target e.g.
is it linear, monotonic or more complex.

.. note::
    We are limited to a small subset of features (less than 3) as it becomes tricky to
    display more than 3/4 variables simultaneously. 
.. maybe put the note later or not at all

The partial dependence function that is plotted is defined as

.. math::
    f_S(x_S) &= \mathbb{E}_{X_{-S}}\left[ f(x_S, X_{-S}) \right]\\
             &= \int f(x_S, x_{-S}) d\mathbb{P}(X_{-S}),

where :math:`X_S` is the set of input features of interest, :math:`X_{-S}` is its complement
and :math:`f(x_S, x_{-S})` is the learned decision function of the model of interest, evalutated
on a sample :math:`x` whose values for the features in S are :math:`x_S` and for features in -S
are :math:`x_{-S}`. The expectation is taken marginally on the values of :math:`X-{-S}`.

The partial dependence function :math:`f_S` is estimated by Monte-Carlo with
:math:`\bar{f}_S` defined as

.. math::
    \bar{f}_S(x_S) \approx \frac{1}{n_\text{n}} \sum_{i=1}^n f(x_S, x_{-S,i}),

where :math:`\{x_{-S,i}}_{i=1}^n` are the values on the training set of the features in -S.
This approximation is equivalent to averaging all the Individual Conditional Expectation (ICE)
curves. These curves are the per-instance version of a PDP, where we display the evolution of
the target when some features change, for one sample of the dataset.
Most plots will include all ICE curves on the same plot with the PDP highlighted.

It is possible to get a measure of feature importance from PDPs, which is explained in this section
of the user guide: :ref:`pdp_importance`.

Example(s)
==========


References
----------
.. bibliography:: ../../tools/references.bib
