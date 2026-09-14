.. _statistical_guarantees:



=============================================
Statistical guarantees for variable selection
=============================================

It is important to determine whether the importance of a given variable is actually different from 0.
A variable importance is typically obtained as a point estimate of the effect of each variable on the outcome of interest.
To gain statistical significance, one  needs instead an estimate of the variable importance distribution.
At the very least, this requires an estimate of the variability of the importance estimate on top of the point estimate itself.
Then, given an estimate of the variable importance distribution,
there exist different characterizations of the statistical significance of variable importance being nonzero,
which correspond to different statistical tests.

False positive rate
--------------------

The p-value of a variable importance measures the probability of observing an importance at least as extreme
as the one obtained, under the null hypothesis that the true importance is zero.
The False Positive Rate or **Type-1 error** rate is the probability of incorrectly rejecting
the null hypothesis for a variable that is actually not important.

Depending on the statistical nature of the data,
one can perform different types of statistical tests to obtain the p-value, such as a t-test or a Wilcoxon test.

Note: quite often, the tests will be carried out on importance values produces by a cross-validation procedure,
which are *not* independent. This makes the standard two-sample tests potentially invalid.
The so-called Nadeau-Bengio correction :footcite:p:`nadeau2003inference` can be used to adjust for this dependency.

The resulting number, the p-value, is valid whenever its distribution under the null hypothesis is dominated
by the uniform distribution on the interval :math:`[0, 1]`.

Example of type-I error control is given in :ref:`sphx_glr_generated_gallery_examples_plot_dcrt_example.py` and
:ref:`sphx_glr_generated_gallery_examples_plot_importance_classification_iris.py`.

In the context of variable importance, the p-value provides a per-feature measure of statistical significance.
However, when multiple features are studied simultaneously,
the probability of observing at least one false positive increases,
necessitating the use of multiple testing corrections such as controlling the Family Wise Error Rate (FWER)
or the False Discovery Rate (FDR).


Family Wise Error Rate (FWER)
------------------------------

The FWER is the probability of making at least one false positive
among all the hypotheses being tested, meaning: declaring that a variable is important while it is not.
Controlling the FWER ensures that the likelihood of any false positive is kept below a specified threshold,
typically using methods such as the Bonferroni correction.
FWER control is typically a stringent criterion,
often leading to conservative results where true positives may be missed in order to minimize false positives.

See :ref:`sphx_glr_generated_gallery_examples_plot_2D_simulation_example.py` for an example of FWER control in practice.

False Discovery Rate (FDR)
---------------------------

The FDR is the expected proportion of false positives among all the hypotheses declared significant.
It estimates the proportion of variables with 0 true importance among those declared significant.
Controlling the FDR allows for a more balanced approach between discovering true positives and limiting false positives,
often using methods such as the Benjamini-Hochberg procedure :footcite:p:`benjamini1995controlling` or knockoff methods :footcite:p:`barber2015controlling`.
FDR control is generally less stringent than FWER control, making it more powerful in situations with many hypotheses,
but it allows for a controlled proportion of false positives.

It is noteworthy that knockoff methods provide control of the FDR,  under some assumptions, but do not give a
per-feature measure of statistical significance.
See :class:`hidimstat.KnockoffInference` for more details on knockoff-based inference.
See :ref:`sphx_glr_generated_gallery_examples_plot_knockoffs_wisconsin.py` for an example of knockoff-based inference in practice.

In any case, it is important to remember that statistical control is tied to some **assumptions** being met.
Non-parametric approaches, such as *permutation tests*, can provide valid statistical control while relying on weaker assumptions.



References
----------

.. footbibliography::
