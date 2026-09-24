.. _high_dimension:


***************************
Inference in high dimension
***************************

.. toctree::
    :maxdepth: 2

Naive inference in high dimension is ill-posed
==============================================

In some cases, data represent high-dimensional measurements of some phenomenon
of interest (e.g. imaging or genotyping). The common characteristic of these
problems is to be very high-dimensional and lead to correlated features. Both
aspects are clearly detrimental to conditional inference, making it both
expensive and powerless:

* Expensive: most learers are quadratic or cubic in the number of features.
  Moreover per-feature inference generally entails a loop over features
* powerless: As dimensionality and correlation increase, it  becomes harder
  and harder to isolate the contribution of each variable, meaning that
  conditional inference is ill-posed.

This is illustrated in the above example, where the Desparsified Lasso
(:class:`hidimstat.DesparsifiedLasso`) struggles
to identify relevant features. We need some data to start:

    >>> from hidimstat._utils.scenario import multivariate_simulation_spatial
    >>>
    >>> n_samples = 75
    >>> shape = (30, 30)
    >>> # size of the edge of the four predictive regions
    >>> roi_size = 4
    >>>
    >>> X_init, y, beta, epsilon = multivariate_simulation_spatial(
    ...     n_samples, shape, roi_size, signal_noise_ratio=10.0, smooth_X=1
    ... )

Then we perform inference on this data using the Desparsified Lasso:

    >>> from hidimstat.desparsified_lasso import DesparsifiedLasso
    >>>
    >>> # compute importance score and associated corrected p-values
    >>> dlasso = DesparsifiedLasso().fit(X_init, y)
    >>> importance = dlasso.importance()
    >>>
    >>> # compute estimated support
    >>>
    >>> import numpy as np
    >>>
    >>> # alpha is the significance level for the statistical test
    >>> alpha = .05
    >>> selected_dl = dlasso.pvalues_ < alpha / (shape[0] * shape[1])
    >>> true_support = beta > 0
    >>> print(f'Desparsified Lasso selected {np.sum(selected_dl * true_support)} features among {np.sum(true_support)}')
    Desparsified Lasso selected 6 features among 64

Feature Grouping and its shortcomings
=====================================

As discussed earlier, feature grouping is a meaningful solution to deal with
such cases: it reduces the number of features to condition on, and generally
also decreases the level of correlation between features.

.. seealso::

   * The :ref:`Grouping documentation <grouping>`


As hinted in :footcite:t:`meinshausen2009pvalues` an efficient way to deal
with such configuration is to take the per-group average of the features:
this leads to a *reduced design*. After inference, all the feature in a given
group obtain the p-value of the group representative. When the inference
engine is Desparsified Lasso, the resulting method is called Clustered
Desparsified lasso, or :class:`hidimstat.ClusterImportance`.

Using the same example as previously, we start by defining a clustering
method that will perform the grouping. For image data, Ward clustering is a
good default model, because it takes into account the neighboring structure
among pixels, which avoids creating overly messy clusters:

    >>> from sklearn.feature_extraction import image
    >>> from sklearn.cluster import FeatureAgglomeration
    >>> from sklearn.linear_model import LassoCV
    >>>
    >>> n_clusters = 50
    >>> connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1])
    >>> ward = FeatureAgglomeration(
    ...     n_clusters=n_clusters, connectivity=connectivity, linkage="ward")
    >>> vim = DesparsifiedLasso(estimator=LassoCV())
    >>>
    >>> # Equipped with this, we can use ClusterImportance:
    >>>
    >>> from hidimstat import ClusterImportance
    >>>
    >>> cludl = ClusterImportance(clustering=ward, vim=vim)
    >>> cludl = cludl.fit(X_init, y)
    >>> importance = cludl.importance(X_init, y)
    >>>
    >>> # compute estimated support
    >>> selected_cdl = cludl.fwer_selection(alpha, n_tests=n_clusters)
    >>> print(f'Clustered Desparsified Lasso selected {np.sum(selected_cdl *  true_support)} features among {np.sum(true_support)}')
    Clustered Desparsified Lasso selected 52 features among 64


Note that inference is also way faster on the compressed representation.

The issue is that  very-high-dimensional data (biological, images, etc.) do
not have any canonical grouping structure. Hence, they rely on grouping
obtained from the data, typically with clustering technique. However, the
resulting clusters bring some undesirable randomness. Think that imputing
slightly different data would lead to different clusters. Since there is no
globally optimal clustering, the wiser solution is to *average* the results
across clusterings. Since it may not be a good idea to average p-values, an
alternative *ensembling* or  *aggregation* strategy is used instead. When the
inference engine is Desparsified Lasso, the resulting method is called
Ensemble of Clustered Desparsified lasso, or :class:`hidimstat.EnsembleImportance`.

The behavior is illustrated here:

    >>> from hidimstat import EnsembleImportance
    >>>
    >>> # ensemble of clustered desparsified lasso (EnsembleImportance)
    >>> encludl = EnsembleImportance(vim=cludl, random_state=0)
    >>> importance = encludl.fit_importance(X_init, y)
    >>> selected_ecdl = encludl.fwer_selection(alpha, n_tests=n_clusters)
    >>> print(f'Ensemble of Clustered Desparsified Lasso selected {np.sum(selected_ecdl *  true_support)} features among {np.sum(true_support)}')
    Ensemble of Clustered Desparsified Lasso selected 0 features among 64

.. topic:: **Full example**

    See the following example for a full file running the analysis:
    :ref:`sphx_glr_generated_gallery_examples_plot_2D_simulation_example.py`

What type of Control does this Ensemble of Clustered inference come with ?
==========================================================================

Ensemble of Clustered Inference is not a local method, so control cannot be
maintained at each brain site in isolation. The notion of a false positive
must be mitigated by the non-local characteristic of the inference performed.
Thus, we introduce the concept of a :math:`\delta`-false positive:
A detection is a delta-false positive if it is at a distance greater than
:math:`\delta` from the support, which is the set of true positives.
Thus, what is controlled is the :math:`\delta`-FWER, i.e., the probability of
reporting a single :math:`\delta`-false positive.
In other words, EnCluDL will likely only report detections at a distance less
than :math:`\delta` from the true support.

What is :math:`\delta` ? It is the diameter of the clusters used in the CluDL
procedure.


The details of the method and the underlying guarantees are described in
:footcite:t:`chevalier2022spatially`


.. topic:: **Other examples**

    See the following example for an application to the analysis of fMRI data:
    :ref:`sphx_glr_generated_gallery_examples_plot_fmri_data_example.py`

    See this example for an illustration on MNIST digit classification:
    :ref:`sphx_glr_generated_gallery_examples_plot_digits.py`


References
==========
.. footbibliography::
