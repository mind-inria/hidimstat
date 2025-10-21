.. _high_dimension:


===========================
Inference in high dimension
===========================

Naive inference in high dimension is ill-posed
----------------------------------------------

In some cases, data represent high-dimensional measurements of some phenomenon of interest (e.g. imaging or genotyping). The common characteristic of these problems is to be very high-dimensional and lead to correlated features. Both aspects are clearly detrimental to conditional inference, making it both expensive and powerless:

* Expensive: most learers are quadratic or cubic in the number of features. Moreover per-feature inference generally entails a loop over features
* powerless: As dimensionality and correlation increase, it  becomes harder and harder to isolate the contribution of each variable, meaning that conditional inference is ill-posed.

This is illustrated in the above example, where the Desparsified Lasso struggles
to identify relevant features. We need some data to start::

    n_samples = 100
    shape = (40, 40)
    n_features = shape[1] * shape[0]
    roi_size = 4  # size of the edge of the four predictive regions

    # generating the data
    from hidimstat._utils.scenario import multivariate_simulation_spatial
    X_init, y, beta, epsilon = multivariate_simulation_spatial(
        n_samples, shape, roi_size, signal_noise_ratio=10., smooth_X=1
    )

Then we perform inference on this data using the Desparsified Lasso::

    from hidimstat.desparsified_lasso import (
    desparsified_lasso,
    desparsified_lasso_pvalue,
    )
    beta_hat, sigma_hat, precision_diagonal = desparsified_lasso(X_init, y)
    _, pval_corr, _, one_minus_pval_corr, cb_min, cb_max = (
        desparsified_lasso_pvalue(X_init.shape[0], beta_hat, sigma_hat, precision_diagonal)
    )
        
    # compute estimated support (second method)
    import numpy as np
    alpha = .05
    selected_dl = np.logical_or(pval_corr < alpha , one_minus_pval_corr < alpha)
    print(f'Desparsified Lasso selected {np.sum(selected_dl)} features among {np.sum(beta > 0)} ')



Feature Grouping and its shortcomings
-------------------------------------

As discussed earlier, feature grouping is a meaningful solution to deal with such cases: it reduces the number of features to condition on, and generally also decreases the level of correlation between features.

.. seealso::

   * The :ref:`Grouping documentation <grouping>`


As hinted in :footcite:t:`meinshausen2009pvalues` an efficient way to deal with such configuration is to take the per-group average of the features: this leads to a *reduced design*. After inference, all the feature in a given group obtain the p-value of the group representative. When the inference engine is Desparsified Lasso, the resulting method is called Clustered Desparsified lasso, or :py:func:`CluDL <hidimstat.clustered_inference>`.

Using the same example as previously, we start by defining a clustering method that will perform the grouping. For image data, Ward clustering is a good default model::

    from sklearn.feature_extraction import image
    from sklearn.cluster import FeatureAgglomeration
    n_clusters = 200 
    connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1])
    ward = FeatureAgglomeration(
        n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
    )

Equipped with this, we can use CluDL::

    from sklearn.preprocessing import StandardScaler
    from hidimstat.ensemble_clustered_inference import (
        clustered_inference,
	clustered_inference_pvalue,
    )
    ward_, beta_hat, theta_hat, omega_diag = clustered_inference(
        X_init, y, ward, n_clusters, scaler_sampling=StandardScaler()
    )
    _, _, pval_corr, _, one_minus_pval_corr = (
        clustered_inference_pvalue(X_init.shape[0], False, ward_, beta_hat, theta_hat, omega_diag)
    )

    # compute estimated support
    selected_cdl = np.logical_or(pval_corr < alpha, one_minus_pval_corr < alpha)
    print(f'Clustered Desparsified Lasso selected {np.sum(selected_cdl)} features among {np.sum(beta > 0)} ')
  
Note that inference is also way faster on the compressed representation.
    
The issue is that  very-high-dimensional data (biological, images, etc.) do not have any canonical grouping structure. Hence, they rely on grouping obtained from the data, typically with clustering technique. However, the resulting clusters bring some undesirable randomness. Think that imputing slightly different data would lead to different clusters. Since there is no globally optimal clustering, the wiser solution is to *average* the results across clusterings. Since it may not be a good idea to average p-values, an alternative *ensembling* or  *aggregation* strategy is used instead. When the inference engine is Desparsified Lasso, the resulting method is called Ensemble of Clustered Desparsified lasso, or :py:func:`EnCluDL<hidimstat.ensemble_clustered_inference>`.

The behavior is illustrated here::

    from hidimstat.ensemble_clustered_inference import (
    ensemble_clustered_inference,
    ensemble_clustered_inference_pvalue,
    )

    # ensemble of clustered desparsified lasso (EnCluDL)
    list_ward, list_beta_hat, list_theta_hat, list_omega_diag = (
        ensemble_clustered_inference(
            X_init,
	    y,
	    ward,
	    n_clusters,
	    scaler_sampling=StandardScaler(),
        ) 
      )
    beta_hat, selected_ecdl = ensemble_clustered_inference_pvalue(
        n_samples,
	False,
	list_ward,
	list_beta_hat,
	list_theta_hat,
	list_omega_diag,
	fdr=fwer_target,
    )
    print(f'Ensemble of Clustered Desparsified Lasso selected {np.sum(selected_ecdl)} features among {np.sum(beta > 0)} ')


.. topic:: **Full example**

    See the following example for a full file running the analysis:
    :ref:`sphx_glr_generated_gallery_examples_plot_2D_simulation_example.py`

What type of Control does this Ensemble of CLustered inference come with ?
--------------------------------------------------------------------------

Ensemble of Clustered Inference is not a local method, so control cannot be maintained at each brain site in isolation.
The notion of a false positive must be mitigated by the non-local characteristic of the inference performed.
Thus, we introduce the concept of a :math:`\delta`-false positive:
A detection is a delta-false positive if it is at a distance greater than $\delta$ from the support, which is the set of true positives.
Thus, what is controlled is the :math:`\delta`-FWER, i.e., the probability of reporting a single false :math:`\delta`-false positive.
In other words, EnCluDL will likely only report detections at a distance less than :math:`\delta` from the true support.

What is :math:`\delta` ? It is the diameter of the clusters used in the CluDL procedure.


The details of the method and the underlying guarantees are described in :footcite:t:`chevalier2022spatially`



References
----------
.. footbibliography::


