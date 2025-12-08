.. _permutation_feature_importance:


==============================
Permutation Feature Importance
==============================

Permutation Feature Importance (PFI) is a model-agnostic approach for quantifying the 
relevance of individual or groups of features in predictive models. It is a 
perturbation-based method that compares the predictive performance of a model on 
unmodified test data—following the same distribution as the training data—
to its performance when the studied feature is marginally permutated. Thus, this approach 
does not require retraining the model contrary to other methods such as :ref:`leave_one_covariate_out`.



Theoretical index
------------------

The target quantity estimated by PFI can be defined as:

.. math::
    \psi^j_{PFI} = \mathbb{E} [\mathcal{L}(y, \mu(X^{\pi(j)}))] - \mathbb{E} [\mathcal{L}(y, \mu(X))],

where :math:`X^{\pi(j)}` corresponds to the input distribution where the :math:`j^{th}` feature
has been replaced by a marginally independent :math:`X^j`. Then, it preserves the
same marginal distribution :math:`P(X_j)` but is independent of the other features and the output.

Note that this method was initially introduced as the mean decrease accuracy (MDA) 
by :footcite:t:`breimanRandomForests2001` for Random Forests. It was initially proposed 
as an heuristic Variable Importance Measure and not as a formal estimator of a 
interesting theoretical quantity. Moreover, it was shown in
:footcite:t:`benard2022SobolMDA` that PFI estimates a quantity that can be decomposed
as the sum of the Total Sobol Index (TSI) :ref:`total_sobol_index` and two extra terms 
that are not significant due to correlations. Thus, the theoretical quantity estimated by PFI is 
not a relevant quantity contrarily to :ref:`leave_one_covariate_out` or 
:ref:`conditional_feature_importance`.


Estimation procedure
--------------------

The estimation of the PFI is relatively simple since there is no need of retraining 
any model as happens with other methods such as :ref:`leave_one_covariate_out` or 
estimating a conditional sampler as in :ref:`conditional_feature_importance`. Since
the distribution from which we are sampling is the marginal distribution of the feature
breaking the relationship with the others, a simple permutation of the feature values
across the individuals is sufficient. Also, note that the same estimated model is 
used for predicting both the original and perturbed datasets.


.. note:: **Extrapolation issues** 

    When features are correlated, permuting a feature can lead to unrealistic
    combinations of feature values that were not observed in the training data.
    This can result in the model making predictions in regions of the feature space
    where it has not been trained, leading to unreliable importance estimates.
    This issue is discussed in :footcite:t:`strobl2008conditional`, 
    :footcite:t:`Hooker2021ExtrapolationPFI`.

    .. figure:: ../generated/gallery/examples/images/sphx_glr_plot_pitfalls_permutation_importance_004.png
        :target: ../generated/gallery/examples/plot_pitfalls_permutation_importance.html
        :align: center


Inference
---------

Doing inference with PFI is challenging due to the extrapolation issues mentioned above.
The distribution of the loss differences is difficult to characterize. Then, obtaining 
valid p-values for the null hypothesis 

.. math::
    \mathcal{H}_0: Y \perp\!\!\!\perp X_j | X_{-j}.
is not straightforward. This leads to many false discoveries, especially when features are correlated.


.. figure:: ../generated/gallery/examples/images/sphx_glr_plot_pitfalls_permutation_importance_002.png
    :target: ../generated/gallery/examples/plot_pitfalls_permutation_importance.html
    :align: center

The conditional version of the PFI (:ref:`conditional_feature_importance`) tackles
both issues of extrapolation and of inference by using conditional sampling instead of
the marginal permutation.

Regression example
------------------
The following example illustrates the use of PFI on a regression task with::

    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from hidimstat import PFI


    >>> X, y = make_regression(n_features=2)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> model = LinearRegression().fit(X_train, y_train)
    
    >>> pfi = PFI(estimator=model)
    >>> pfi = pfi.fit(X_train, y_train)
    >>> features_importance = pfi.importance(X_test, y_test)


Classification example
----------------------
To measure feature importance in a classification task, a classification loss should be
used, in addition, the prediction method of the estimator should output the corresponding 
type of prediction (probabilities or classes). The following example illustrates the use
of PFI on a classification task::

    >>> from sklearn.datasets import make_classification
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.metrics import log_loss
    >>> from sklearn.model_selection import train_test_split
    >>> from hidimstat import PFI

    >>> X, y = make_classification(n_features=4)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> model = RandomForestClassifier().fit(X_train, y_train)
    >>> pfi = PFI(
    ...     estimator=model,
    ...     loss=log_loss,
    ...     method="predict_proba",
    ... )
    >>> pfi = pfi.fit(X_train, y_train)
    >>> features_importance = pfi.importance(X_test, y_test)

References
----------
.. footbibliography::
