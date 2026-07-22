.. _shapley_additive_global_explanation:


==========================================
Shapley Additive Global Explanation (SAGE)
==========================================

Introduced in :footcite:t:`Covert2020`


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
