.. _total_sobol_index:

===============================
Generalized Total Sobol Index
===============================

The Generalized Total Sobol Index (TSI) is a theoretical measure of feature importance
that comes from sensitivity analysis. It has gained popularity in the field of interpretable
Machine Learning (:footcite:t:`benard2022SobolMDA`, :footcite:t:`Williamson_General_2023`)
as an interesting target to assess the relevance of features in predictive models.
Indeed, the TSI can be intuitively defined by the performance drop of a predictive model
when a feature is removed, similarly to an ablation study. More formally, it can
be defined as: 

.. math::
    \psi^j_{TSI} = \mathbb{E} \left[\mathcal{L}\left(y, \mu(X)\right)\right] - \mathbb{E} \left[\mathcal{L}\left(y, \mu_{-j}(X^{-j})\right)\right],

where :math:`X^{-j}` denotes the feature vector without the :math:`j^{th}` feature, and
:math:`\mu_{-j}(X^{-j})` is the same predictive model as :math:`\mu(X)` but retrained 
on the reduced feature set :math:`X^{-j}`. When :math:`\mathcal{L}` is the squared loss, 
for a regression task, :math:`\mu_{-j}(X^{-j}) = \mathbb{E}[y \mid X^{-j}]` and when 
:math:`\mathcal{L}` is the log-loss, for a classification task, :math:`\mu_{-j}(X^{-j}) = P(y \mid X^{-j})`.

It is defined as generalized TSI since the original Sobol indices were introduced
in the context of variance-based sensitivity analysis (:footcite:t:`sobol1993sensitivity`) 
which corresponds to having a quadratic loss. 

In general, it can be estimated directly using a plug-in estimator in the definition,
which gives the Leave-One-Covariate-Out (:ref:`leave_one_covariate_out`).


Mean Squared Error (MSE) case
-------------------------------

In the regression setting with the MSE loss, the TSI can be expressed in several
equivalent forms which provide different intuitions of why the target quantity is relevant:


.. math::

   \begin{align}
   \psi^j_{\mathrm{TSI}}
   &= \mathbb{E}\!\left[\left(\mu_{-j}(X^{-j})- Y\right)^2\right]
      - \mathbb{E}\!\left[(\mu(X)- Y)^2\right]
      &\textcolor{gray}{\text{loss / refitting}}\\
   &= \sigma^2(R^2_{-j}-R^2)
      &\textcolor{gray}{\text{adjustment}}\\
   &= \mathbb{E}\!\left[\mathrm{Var}(Y\mid X^{-j})\right]
      &\textcolor{gray}{\text{variance}}\\
   &= \mathbb{E}\!\left[\left(\mathbb{E}[\mu(X)\mid X^{-j}]- Y\right)^2\right]
      - \mathbb{E}\!\left[(\mu(X)- Y)^2\right]
      &\textcolor{gray}{\text{marginalization}}\\
   &= \frac{1}{2}\!\left[\mathbb{E}\!\left[\left(\mu(X^{\pi(j\mid -j)})- Y\right)^2\right]
      - \mathbb{E}\!\left[(\mu(X)- Y)^2\right]\right]
      &\textcolor{gray}{\text{perturbation}}.
   \end{align}


The first formulation (loss / refitting) provides the definition of the TSI with
the quadratic loss. The second formulation (adjustment) shows that the TSI can be seen
as the dropped R-squared when removing the :math:`j^{th}` feature, scaled by the output variance. 
The third formulation (variance) shows that it quantifies the variance of the output
explained uniquely by the :math:`j^{th}` feature. The fourth formulation (marginalization)
shows that it can be seen as the performance drop when marginalizing the contribution of the
:math:`j^{th}` feature in the predictive model. Finally, the last formulation (perturbation)
shows that it can be seen as the dropped performance by perturbing the :math:`j^{th}` feature 
conditionally on the rest. 

Also, we observe that LOCO (:ref:`leave_one_covariate_out`), CFI (:ref:`conditional_feature_importance`)
and conditional SAGE value functions (:footcite:t:`Covert2020`)  target the TSI since they
consists on plug-in estimators of the refitting, perturbation and marginalization formulation respectively.

Cross-entropy case
--------------------

In the classification setting with the cross-entropy (log-loss), the TSI can be expressed 
in terms of information-theoretic quantities:

.. math::

   \begin{align}
   \psi_{\mathrm{TSI}}^j
   &= \mathrm{I}(Y; X^j \mid X^{-j}) &\textcolor{gray}{\text{mutual information}}
   \\&= D_{\mathrm{KL}}\left(P\left(Y, X^j \mid X^{-j}\right) \,\big\|\, P\left(Y \mid X^{-j}\right) P\left(X^j \mid X^{-j}\right)\right)&\textcolor{gray}{\text{KL divergence}}.
   \end{align}

From the first formulation (mutual information), we see that the TSI quantifies the
mutual information between the output :math:`Y` and the feature :math:`X^j` conditionally on the
rest of the features :math:`X^{-j}`. Thus, it quantifies how much information
about :math:`Y` is contained in :math:`X^j` that is not already contained in :math:`X^{-j}`.
The second formulation (Kullback-Leibler divergence) shows that the TSI can be seen as how much 
the original probability distribution diverges from the one that we would have obtained 
if the :math:`j^{th}` feature was conditionally independent from the output given the rest of the features.


References
----------

.. footbibliography::


