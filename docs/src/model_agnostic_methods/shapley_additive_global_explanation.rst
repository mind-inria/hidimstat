.. _shapley_additive_global_explanation:


==========================================
Shapley Additive Global Explanation (SAGE)
==========================================

Introduced in :footcite:t:`Covert2020`, Shapley Additive Global Explanation
(SAGE) builds on top of Shapley values. In game theory, Shapley values have been
proposed as a solution to the credit allocation scheme for cooperative games.
As explained in the `Shapley value wikipedia page <https://en.wikipedia.org/wiki/Shapley_value>`_:

    The Shapley value determines each player's contribution by considering how
    much the overall outcome changes when they join each possible combination of
    other players, and then averaging those changes.

Shapley values have then been applied to machine learning models to measure the
contribution of features. The definition above can be translated to the
predictive modeling setting by replacing the notion of "outcome" by the model
performance measured through a loss function, and the notion of "players" by
features.


.. figure:: ../generated/gallery/examples/images/sphx_glr_plot_msage_001.png
    :target: ../generated/gallery/examples/plot_msage.html
    :align: center


Theoretical index
------------------



.. math::
    \psi^j_{SAGE} = \frac{1}{d}\sum_{S\subseteq [d] \backslash {j}} \binom{d-1}{|S|}^{-1}\left[v_{f_\star}(S\cup \{j\}) - v_{f_\star}(S)\right]



Where in regression, :math:`\mu_{-j}(X^{-j}) = \mathbb{E}[Y| X^{-j}]` is the
theoretical model without the :math:`j^{th}` feature.

Estimation procedure
--------------------

TODO


.. note:: **This** is a note

    todo


Inference
---------


Regression example
------------------
The following example illustrates the use of SAGE on a regression task with::

    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from hidimstat import SAGE


    >>> X, y = make_regression(n_features=2)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> model = LinearRegression().fit(X_train, y_train)

    >>> cfi = CFI(estimator=model, imputation_model_continuous=LinearRegression())
    >>> cfi = cfi.fit(X_train, y_train)
    >>> features_importance = cfi.importance(X_test, y_test)


Classification example
----------------------
TODO

References
----------
.. footbibliography::
