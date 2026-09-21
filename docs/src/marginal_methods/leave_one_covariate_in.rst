.. _leave_one_covariate_in:

Leave-One-Covariate-In
======================

Leave One Covariate In (LOCI) is a model-agnostic approach for quantifying the
relevance of individual or groups of features in predictive models. It is a
refitting-based method that compares the predictive performance of the empty
model with performance of a model refitted just with the feature(s) of interest.


Theoretical index
------------------

Leave One Covariate In (LOCI) is a model-agnostic method for estimating feature
importance through refitting. The goal is to measure how predictive the model
would be when accommodating only the given feature. The importance of feature
:math:`j` is then quantified by the resulting gain in model performance when the
model is refitted using only that feature.

.. math::
\psi^j_{\mathrm{LOCI}}
  = \mathbb{E}\left[\mathcal{L}\left(Y, \mu_{\emptyset}(X^{\emptyset})\right)\right]
  - \mathbb{E}\left[\mathcal{L}\left(Y, \mu_j(X^j)\right)\right].

Here, :math:`\mu_{\emptyset}(X^{\emptyset})` denotes the theoretical model
without any features (e.g., the mean of :math:`Y` when the loss is the mean
squared error), while :math:`\mu_j(X^j)` denotes the theoretical model using
only the :math:`j`-th feature.

Thus, in contrast to LOCO, which aims to quantify the performance drop resulting
from removing feature :math:`j` while retaining all other features, LOCI studies
the performance gain obtained when feature :math:`j` is used as the only
predictor. The target quantity estimated by LOCI is therefore closely related
to the first-order Sobol index, which corresponds to the fraction of the
variance of the output that can be explained by the studied feature alone.

Indeed, under the quadratic loss, we have

.. math::
\begin{aligned}
\psi^j_{\mathrm{LOCI}}
&= \mathbb{E}\left[
\mathcal{L}\left(Y,\mu_{\emptyset}(X^{\emptyset})\right)
\right]
-
\mathbb{E}\left[
\mathcal{L}\left(Y,\mu_j(X^j)\right)
\right] \
&= \operatorname{Var}\left(\mathbb{E}[Y\mid X^j]\right) \
&= \operatorname{Var}(Y) R_j^2,
\end{aligned}

where :math:`R_j^2` denotes the coefficient of determination obtained by
predicting :math:`Y` from :math:`X^j` alone. Consequently, normalizing the LOCI
quantity by the total variance of :math:`Y` gives

.. math::
\frac{\psi^j_{\mathrm{LOCI}}}{\operatorname{Var}(Y)}
=
\frac{\operatorname{Var}\left(\mathbb{E}[Y\mid X^j]\right)}
{\operatorname{Var}(Y)}
=
S_j,

where :math:`S_j` is the first-order Sobol index of feature :math:`j`.




Estimation procedure
--------------------

The estimation of LOCI consists of a direct plug-in estimate of
:math:`\psi^j_{\mathrm{LOCI}}`. It involves refitting the model using only the
feature(s) of interest and comparing its predictive performance with that of
the constant model. This can be done using cross-validation or a hold-out test
set to evaluate the performance of both models.

Alternatively, the target quantity can be estimated through conditional
marginalization, which gives rise to the conditional SAGE value functions
(:footcite:t:`reyerolobo2025principledapproachcomparingvariable`).

Inference
---------
Under standard assumptions, such as an additive model
:math:`Y = \mu(X) + \epsilon`, Leave One Covariate In (LOCI) can be used for
marginal independence testing to determine whether a feature provides any
information about the response. In other words, we test whether the response
is independent of the feature under consideration:

.. math::
  \mathcal{H}_0: Y \perp!!!\perp X^j.

The core of this inference is to test the statistical significance of the loss
difference estimated by LOCI. Consequently, a one-sample test on the loss
differences, or equivalently a paired test on the corresponding losses, can be
performed.

It is important to note that the null hypothesis of marginal independence
detects whether a feature is associated with the response on its own. Thus, a
feature may be identified as important simply because it is correlated with
other features that are predictive of the response. In contrast, testing
conditional independence allows us to determine whether a feature provides
additional information about the response after accounting for the effect of
the other features.

Regression example
------------------
The following example illustrates the use of LOCI on a regression task with::

    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from hidimstat import LOCI


    >>> X, y = make_regression(n_features=2)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> model = LinearRegression().fit(X_train, y_train)

    >>> loci = LOCI(estimator=model)
    >>> loci = loci.fit(X_train, y_train)
    >>> features_importance = loci.importance(X_test, y_test)


Classification example
----------------------
To measure feature importance in a classification task, a classification loss should be
used, in addition, the prediction method of the estimator should output the corresponding
type of prediction (probabilities or classes). The following example illustrates the use
of LOCI on a classification task::

    >>> from sklearn.datasets import make_classification
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.metrics import log_loss
    >>> from sklearn.model_selection import train_test_split
    >>> from hidimstat import LOCI

    >>> X, y = make_classification(n_features=4)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> model = RandomForestClassifier().fit(X_train, y_train)
    >>> loci = LOCI(
    ...     estimator=model,
    ...     loss=log_loss,
    ...     method="predict_proba",
    ... )
    >>> loci = loci.fit(X_train, y_train)
    >>> features_importance = loci.importance(X_test, y_test)

References
----------
.. footbibliography::
