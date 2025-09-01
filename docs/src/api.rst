.. _api_documentation:

=================
API
=================

Estimators
==========

.. currentmodule:: hidimstat

Functions
=========

.. autosummary::
   :toctree: ./generated/api/function/
   :template: function.rst

   quantile_aggregation
   clustered_inference
   clustered_inference_pvalue
   desparsified_lasso
   desparsified_lasso_pvalue
   desparsified_group_lasso_pvalue
   ensemble_clustered_inference
   ensemble_clustered_inference_pvalue
   model_x_knockoff
   reid

Classes
=======

.. autosummary::
   :toctree: ./generated/api/class/
   :template: class.rst
   
   BaseVariableImportance
   BasePerturbation
   LOCO
   CRT
   ConditionalRandimizationTest
   CFI
   PFI
   D0CRT

Helper Classes
==============
.. autosummary::
   :toctree: ./generated/api/class/
   :template: class.rst

   statistical_tools.gaussian_distribution.GaussianDistribution

Helper Functions
================
.. autosummary::
   :toctree: ./generated/api/class/
   :template: function.rst
   
   statistical_tools.lasso_test.lasso_statistic

