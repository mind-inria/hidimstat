.. _desparsified_lasso:


======================
Desparsified Lasso
======================

Desparsified Lasso, also known as debiased Lasso, is a method that aims at estimating 
the regression coefficients :footcite:t:`hastie2015statistical`. To do so, it uses coefficients obtained from a Lasso 
regression and corrects the bias induced by the L1-penalty. This method is particularly
useful in high-dimensional settings where the number of features exceeds the number of 
samples.

Target quantity
---------------

This method assumes a linear model of the form: :math:`Y = X \beta^\star + \epsilon`, where
:math:`Y` is the response variable, :math:`X` is the design matrix, :math:`\beta` is
the vector of regression coefficients, and :math:`\epsilon` is an additive noise term.
The target quantity estimated by Desparsified Lasso is the vector of regression
coefficients :math:`\beta^\star`. Denoting :math:`\hat{\beta}_\lambda` the Lasso estimator
with regularization parameter :math:`\lambda`, and :math:`\Theta` an approximate inverse 
of the empirical covariance matrix :math:`\hat{\Sigma} = \frac{1}{n} X^T X`, the 
Desparsified Lasso estimator is given by:

.. math::

    \hat{\beta}_{DL} = \hat{\beta}_\lambda + \frac{1}{n} \Theta X^T (Y - X \hat{\beta}_\lambda)


Estimation procedure
--------------------

The provided implementation :class:`hidimstat.DesparsifiedLasso` is based on the one 
proposed by :footcite:t:`zhang2014confidence`. It first fits a Lasso regression to obtain the initial
estimator :math:`\hat{\beta}_\lambda`, and then estimates nodewise Lasso regressions, 
each predicting a feature :math:`X^j` using all other features :math:`X^{-j}`. The 
residuals of these regressions, denoted :math:`z^j` which intuitively capture the part 
of :math:`X^j` that is not explained by the other features, are used to construct the 
debiased estimate: 

.. math::

    \hat{\beta}_{DL}^j = \frac{z^j Y}{(z^j)^\top X^j} - \sum_{k \neq j} \frac{(z^{j})^\top X^k \hat{\beta}_\lambda^k}{(z^j)^\top X^j}

where :math:`\hat{\beta}_{DL}^j` is the :math:`j`-th component of the Desparsified 
Lasso estimate. 


Inference 
----------

As described in :footcite:t:`chevalier2021decoding`, under some sparsity assumptions, 
it can be obtained that the Desparsified Lasso estimator is asymptotically normal. This 
property allows to derive confidence intervals and p-values for the regression
coefficients. 


Extensions to spatially structured data
---------------------------------------

When dealing with spatially structured data, such as neuroimaging data where features
correspond to voxels in a 3D brain image, and there is a natural spatial correlation between
neighboring voxels, the standard Desparsified Lasso, which treats each feature
independently, may not fully leverage the spatial structure present in the data. To 
mitigate this, one can incorporate spatial information into the estimation process as 
suggested by :footcite:t:`chevalier2021decoding`. This is achieved by performing inference 
on clusters, instead of individual features (see :class:`hidimstat.CluDL`). Such clusters 
can either be defined a priori based on anatomical knowledge (using atlases), or can be 
obtained through data-driven clustering methods. Additionally, when using data-driven
clustering, multiple random clusterings can be aggregated to improve the stability, as 
implemented in :class:`hidimstat.EnCluDL`.


.. figure:: ../generated/gallery/examples/images/sphx_glr_plot_fmri_data_example_001.png
    :target: ../generated/gallery/examples/plot_fmri_data_example.html
    :align: center