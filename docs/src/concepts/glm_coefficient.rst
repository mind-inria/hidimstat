.. _glm_coefficient:


===========================================
Generalized Linear Model (GLM) Coefficient
===========================================

Assuming that the data is generated from a Linear Model,

.. math::
    Y = \sum \beta_j X^j+ \epsilon,
    \quad \text{ with }\varepsilon\sim\mathcal{N}(0, \sigma^2),

the most natural way of defining the importance of each feature is through the absolute/squared value
of the corresponding coefficient :math:`\beta_j` :footcite:`verdinelli2024feature`:

.. math::
    \psi^j_{GLM} = \beta_j^2.

This definition has the advantage of being simple and interpretable. Moreover, it
can be directly extended to classification for example by considering the
coefficients of a Generalized Linear Model (GLM).


.. note:: **Misspecification of the underlying distribution**

    This definition relies on the strong assumption that the data is generated
    from a (Generalized) Linear Model. In practice, this assumption is often violated,
    which can lead to misspecifying the underlying data distribution and therefore
    to misleading importance measures (:footcite:`Molnar2022Pitfalls`). See
    :ref:`sphx_glr_generated_gallery_examples_plot_model_agnostic_importance.py`.


They can be estimated using standard statistical techniques for fitting GLMs.
There are standard penalization procedures, such as Lasso or Ridge, that address
estimation issues in high-dimensional or highly correlated settings. Moreover,
there are debiased alternatives to these procedures (see :ref:`desparsified_lasso`).

Behind this simple idea lie several core procedures of statistical inference.
For instance, the popular Model-X Knockoff (:ref:`knockoffs`, :footcite:`candes2018panning`)
and the Distilled Conditional Randomization Test
(:ref:`d0crt`, :footcite:`liu2022fast`) usually rely on GLM coefficients to perform
variable selection with FDR and type-I error control, respectively.




References
----------

.. footbibliography::