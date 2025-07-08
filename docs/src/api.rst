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
   
   LOCO
   CPI
   PFI
   D0CRT

Base Classes
============

.. autosummary::
   :toctree: ./generated/api/base/
   :template: class.rst

   BaseVariableImportance
   BaseVariableImportanceGroup
   BasePerturbation


Marginal Importance
===================
.. autosummary::
   :toctree: ./generated/api/marginal
   :template: class.rst

   LOCI
   LeaveOneCovariateIn