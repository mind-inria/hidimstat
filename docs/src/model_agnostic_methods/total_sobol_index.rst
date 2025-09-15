.. _total_sobol_index:


======================
Total Sobol Index
======================

The Total Sobol Index (TSI) :footcite:t:`sobol1993sensitivity` is a widely used measure of feature importance that is
rooted in sensitivity analysis. It can intuitively be defined by the performance drop
of a predictive model when a feature is removed, similarly to an ablation study. For a
data generating process :math:`y = \mu(X) + \epsilon`, the TSI can be expressed as:

.. math::
    \psi_j^{TSI} = \mathbb{E} \left[\mathcal{L}\left(y, \mu(X)\right)\right] - \mathbb{E} \left[\mathcal{L}\left(y, \mu_{-j}(X^{-j})\right)\right],

where :math:`X^{-j}` denotes the feature vector without the :math:`j^{th}` feature, and
:math:`\mu_{-j}(X^{-j}) = \mathbb{E}[y | X^{-j}]` is the "sub-model".