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
to identify relevant features::

    n_samples = 100
    shape = (40, 40)
    n_features = shape[1] * shape[0]
    roi_size = 4  # size of the edge of the four predictive regions

    # generating the data
    from hidimstat._utils.scenario import multivariate_simulation_spatial
    X_init, y, beta, epsilon = multivariate_simulation_spatial(
    n_samples, shape, roi_size, signal_noise_ratio=10., smooth_X=1
    )

    from hidimstat.desparsified_lasso import (
    desparsified_lasso,
    desparsified_lasso_pvalue,
    )
    beta_hat, sigma_hat, precision_diagonal = desparsified_lasso(X_init, y)
    pval, pval_corr, one_minus_pval, one_minus_pval_corr, cb_min, cb_max = (
    desparsified_lasso_pvalue(X_init.shape[0], beta_hat, sigma_hat, precision_diagonal)
    )
    
    # compute estimated support (first method)
    from hidimstat.statistical_tools.p_values import zscore_from_pval
    zscore = zscore_from_pval(pval, one_minus_pval)
    selected_dl = zscore > thr_nc  # use the "no clustering threshold"
    
    # compute estimated support (second method)
    selected_dl = np.logical_or(
    pval_corr < fwer_target / 2, one_minus_pval_corr < .05
    )
    print(f'Desparsified Lasso selected {np.sum(selected_dl)} features')
    print(f'among {np.sum(beta_hat > 0)} ')

.. topic:: **Full example**

    See the following example for a full file running the analysis:
    :ref:`sphx_glr_generated_gallery_examples_plot_2D_simulation_example.py`


Feature Grouping and its shortcomings
-------------------------------------

As discussed earlier, feature grouping is a meaningful solution to deal with such cases: it reduces the number of features to condition on, and generally also decreases the level of correlation between features (XXX see grouping section).
As hinted in [Meinshausen XXX] an efficient way to deal with such configuration is to take the per-group average of the features: this leads to a *reduced design*. After inference, all the feature in a given group obtain the p-value of the group representative. When the inference engine is Desparsified Lasso, the resulting mùethod is called Clustered Desparsified lasso, or **CluDL**.

The issue is that  very-high-dimensional data (biological, images, etc.) do not have any canonical grouping structure. Hence, they rely on grouping obtained from the data, typically with clustering technique. However, the resulting clusters bring some undesirable randomness. Think that imputing slightly differnt data would lead to different clusters. Since there is no globally optimal clustering, the wiser solution is to *average* the results across clusterings. Since it may not be a good idea to average p-values, an alternative *ensembling* or  *aggregation* strategy is sued instead. When the inference engine is Desparsified Lasso, the resulting mùethod is called Ensemble of Clustered Desparsified lasso, or **EnCluDL**.

Example
-------



.. topic:: **Full example**

    See the following example for a full file running the analysis:
    :ref:`sphx_glr_generated_gallery_examples_plot_2D_simulation_example.py`

What type of Control does this Ensemble of CLustered inference come with ?
--------------------------------------------------------------------------

