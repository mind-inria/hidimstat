
.. _pdp_importance:

============================
PDP-based feature importance
============================

A Partial Dependence Plot (PDP) :footcite:`friedman2001greedy` is a tool used to visualize 
the dependence between the target variable and a feature (or set of features) of interest.
To see how they are used to understand the relationship between features and the target,
see :ref:`partial_dependance_plots`. Here we will see how a marginal measure of feature
importance can be derived from PDPs.

A PDP is a plot of the partial dependence function that defined as

.. math::
    f_S(x_S) &= \mathbb{E}_{X_{-S}}\left[ f(x_S, X_{-S}) \right]\\
             &= \int f(x_S, x_{-S}) d\mathbb{P}(X_{-S}),

and estimated with 

.. math::
    \bar{f}_S(x_S) \approx \frac{1}{n_\text{n}} \sum_{i=1}^n f(x_S, x_{-S,i}).

It will show how much the value of the target changes when a (or multiple) feature varies. Intuitively, the
flatter the PDP, the less importance a feature should have as it appears to have little impact
on the target. On the other hand, the more a PDP varies, the more signal on the target should
be present in the feature.
Greenwell, Boehmke, and McCarthy :footcite:`greenwell2018simple` propose the following measure
of feature importance for regression:

.. math::
    \Psi^{PDP}_S = \sqrt{ \frac{1}{K-1} \sum_{k=1}^K (\bar{f}_S(x_S^k) - \frac{1}{K} \sum_{k=1}^K \bar{f}_S(x_S^k))^2 }.

It corresponds to the deviation of each unique feature value from the average curve.

In classification they suggest:

.. math::
    \Psi^{PDP}_S = \frac{ \max_k(\bar{f}_S(x_S^k)) - \min_k(\bar{f}_S(x_S^k)) }{4}.

It is an estimation of the deviation based on the range of values, based on the fact that
for the normal distribution, roughly 95% of the of the data are bewteen minus two and plus
two standard deviations. Therefore the range divided by four is a rough estimation of the
deviation.


References
---------
.. footbibliography::

