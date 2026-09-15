.. _d0crt:

Distilled Conditional Randomization Test
========================================

The Distilled Conditional Randomization Test (dCRT) is a method for variable selection
that tests whether each feature :math:`X^j` is conditionally independent of the target
:math:`Y` given all other features :math:`X^{-j}`. It was introduced by
:footcite:t:`liu2022fast` as a computationally efficient version of the Conditional
Randomization Test (CRT) from :footcite:t:`candes2018panning`. The key idea is to
replace the expensive resampling step of the CRT with a *distillation* procedure that
decomposes the problem into simple residual computations.


Target quantity
---------------

The dCRT tests the conditional independence hypothesis for each feature :math:`j`:

.. math::
    \mathcal{H}_0^j: Y \perp\!\!\!\!\perp X^j \mid X^{-j}.

Unlike :ref:`Model-X Knockoffs <knockoffs>`, which control the False Discovery
Rate (FDR) across the full set of selected variables, the dCRT produces a p-value for
each individual feature. This allows conditional independence testing at the
single-feature level with type-I error control.

Under a linear model :math:`Y = X \beta^* + \varepsilon`, the null hypothesis is
equivalent to testing :math:`\beta^*_j = 0`. The test statistic produced by dCRT follows
a standard normal distribution under the null, enabling direct p-value computation
without permutations.


Estimation procedure
--------------------

The original CRT (:footcite:t:`candes2018panning`) requires, for each feature, fitting
the predictive model many times on resampled data to build the null distribution of the
test statistic. This makes it prohibitively expensive when the number of features is
large. The dCRT replaces this resampling loop with a *distillation* procedure: instead
of repeatedly refitting the model, it reduces the problem to computing residuals from
two regressions. For each feature :math:`j`:

**1. Distillation.** The key idea is to decompose the problem into two regression
sub-problems that remove the effect of all other features:

- *X-distillation*: Regress :math:`X^j` on :math:`X^{-j}` to obtain the residual
  :math:`\hat{\epsilon}_j^X = X^j - \hat{\nu}_j(X^{-j})`. This residual captures the
  part of :math:`X^j` that cannot be predicted from the other features.
- *Y-distillation*: Regress :math:`Y` on :math:`X^{-j}` to obtain the residual
  :math:`\hat{\epsilon}_j^Y = Y - \hat{\mu}_{-j}(X^{-j})`. This residual captures the
  part of :math:`Y` that cannot be predicted from the other features.

While the X-distillation step currently uses a linear model, the Y-distillation model
can be any supervised learner (e.g., random forests, gradient boosting). This makes the
dCRT applicable beyond purely linear settings: the test remains valid as long as the
X-distillation model is correctly specified.

**2. Test statistic.** The test statistic is the normalized correlation between the two
residuals:

.. math::
    T_j = \frac{\hat{\epsilon}_j^{Y\top} \hat{\epsilon}_j^X}
    {\sqrt{n \, \hat{\sigma}_j^2 \, \|\hat{\epsilon}_j^Y\|^2 / n}}

where :math:`\hat{\sigma}_j^2` is the estimated variance of the X-residuals. Under the
null hypothesis, :math:`T_j \sim \mathcal{N}(0, 1)` asymptotically.

.. note:: **Screening (optional)**

    When the number of features is large, an optional screening step can be enabled via
    the ``lasso_screening`` parameter. A Lasso regression is fitted on the full data and
    features with coefficients below a percentile threshold are discarded before running
    the distillation. This reduces computation but can be turned off by setting
    ``lasso_screening=None``.

.. note:: **Logistic regression variant (dCRT-logit)**

    When the estimator is a logistic regression, :class:`hidimstat.D0CRT` automatically
    switches to the dCRT-logit approach from :footcite:t:`nguyen2022conditional`, which
    adapts the distillation and test statistic to the logistic loss. The null distribution
    remains standard normal. See the
    :ref:`dCRT-logit example <sphx_glr_generated_gallery_examples_plot_dcrt_logit.py>`
    for an illustration.


Inference
---------

Since the test statistic :math:`T_j` follows a standard normal distribution under the
null, p-values are obtained directly as
:math:`p_j = 2 \, \Phi(-|T_j|)`, where :math:`\Phi` is the standard normal CDF.

These p-values can then be used for variable selection with type-I error control
(via ``pvalue_selection``) or FDR control (via ``fdr_selection``).


Regression example
------------------
The following example illustrates the use of dCRT on a regression task::

    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import LassoCV
    >>> from hidimstat import D0CRT

    >>> X, y = make_regression(n_samples=200, n_features=20, n_informative=5)

    >>> dcrt = D0CRT(estimator=LassoCV(), random_state=0)
    >>> dcrt.fit(X, y)
    >>> features_importance = dcrt.importance(X, y)

    >>> # Selection based on p-value threshold
    >>> selected_features = dcrt.pvalue_selection(threshold_max=0.05)


Classification example
-----------------------
For classification with logistic regression, the dCRT-logit variant is used
automatically::

    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegressionCV
    >>> from hidimstat import D0CRT

    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 20))
    >>> beta = np.zeros(20)
    >>> beta[:2] = 10.0
    >>> y = rng.binomial(1, 1 / (1 + np.exp(-X @ beta)))

    >>> dcrt = D0CRT(
    ...     estimator=LogisticRegressionCV(penalty="l1", solver="liblinear"),
    ...     lasso_screening=LogisticRegressionCV(penalty="l1", solver="liblinear"),
    ...     random_state=0,
    ... )
    >>> dcrt.fit(X, y)
    >>> features_importance = dcrt.importance(X, y)


Examples
--------

.. minigallery:: hidimstat.D0CRT

References
==========

.. footbibliography::
