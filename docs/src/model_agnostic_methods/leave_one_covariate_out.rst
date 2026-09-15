.. _leave_one_covariate_out:


========================
Leave-One-Covariate-Out
========================

Leave One Covariate Out (LOCO) is a model-agnostic approach for quantifying the
relevance of individual or groups of features in predictive models. It is a
refitting-based method that compares the predictive performance of the original
model with performance of a model refitted without the feature(s) of interest.


Theoretical index
------------------

Leave One Covariate Out (LOCO) is a model-agnostic method for estimating feature
importance through refitting. The goal is to measure how predictive the model
would have been if a given feature had not been used. The importance of feature
:math:`j` is then quantified by the resulting drop in model performance when the
model is refitted without that feature.

.. math::
    \psi^j_{LOCO} = \mathbb{E} [\mathcal{L}(Y, \mu_{-j}(X^{-j}))] - \mathbb{E} [\mathcal{L}(Y, \mu(X))].

The target quantity estimated by LOCO is the Total Sobol Index (TSI) :ref:`total_sobol_index`.

Estimation procedure
--------------------

The estimation of LOCO consists of a direct plug-in in the Total Sobol Index :ref:`total_sobol_index`. It
involves refitting the model without the feature(s) of interest and comparing
the performance of the original model with that of the refitted model.
This can be done using cross-validation or a hold-out test set to evaluate the performance of both models.



Inference
---------
Under standard assumptions such as additive model: :math:`Y = \mu(X) + \epsilon`,
Leave One Covariate Out (LOCO) allows for conditional independence testing, which
determines if a feature provides any unique information to the model's predictions that
isn't already captured by the other features. Essentially, we are testing whether the output is independent from the studied feature given the rest of the input:

.. math::
    \mathcal{H}_0: Y \perp\!\!\!\perp X^j | X^{-j}.


The core of this inference is to test the statistical significance of the loss
differences estimated by LOCO. Consequently, a one-sample test on the loss differences
(or a paired test on the losses) needs to be performed.

Two technical challenges arise in this context:

* When cross-validation (for instance, k-fold) is used to estimate LOCO, the loss
  differences obtained from different folds are not independent. Consequently,
  performing a simple t-test on the loss differences is not valid. This issue can be
  addressed by a corrected t-test accounting for this dependence, such as the one
  proposed in :footcite:t:`nadeau1999inference`.
* Vanishing variance: Under the null hypothesis, even if the loss difference
  converges to zero, the variance of the loss differences also vanishes due to the quadratic
  functional (:footcite:t:`Williamson_General_2023`) . This makes the standard one-sample
  t-test invalid. This second issue can be handled by correcting the variance estimate
  or using other nonparametric test.


Regression example
------------------
The following example illustrates the use of LOCO on a regression task with::

    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from hidimstat import LOCO


    >>> X, y = make_regression(n_features=2)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> model = LinearRegression().fit(X_train, y_train)

    >>> loco = LOCO(estimator=model)
    >>> loco = loco.fit(X_train, y_train)
    >>> features_importance = loco.importance(X_test, y_test)


Classification example
----------------------
To measure feature importance in a classification task, a classification loss should be
used, in addition, the prediction method of the estimator should output the corresponding
type of prediction (probabilities or classes). The following example illustrates the use
of LOCO on a classification task::

    >>> from sklearn.datasets import make_classification
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.metrics import log_loss
    >>> from sklearn.model_selection import train_test_split
    >>> from hidimstat import LOCO

    >>> X, y = make_classification(n_features=4)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> model = RandomForestClassifier().fit(X_train, y_train)
    >>> loco = LOCO(
    ...     estimator=model,
    ...     loss=log_loss,
    ...     method="predict_proba",
    ... )
    >>> loco = loco.fit(X_train, y_train)
    >>> features_importance = loco.importance(X_test, y_test)

References
----------
.. footbibliography::
