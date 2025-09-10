.. _api_documentation:

=================
API
=================

.. currentmodule:: hidimstat

Base Classes
============

.. autosummary::
   :toctree: ./generated/api/base/
   :template: class.rst

   base_variable_importance.BaseVariableImportance
   base_perturbation.BasePerturbation

Feature Importance Classes
========================

.. autosummary::
   :toctree: ./generated/api/class/
   :template: class.rst
   
   LOCO
   CFI
   PFI
   D0CRT

Feature Importance functions
===========================

.. autosummary::
   :toctree: ./generated/api/class/
   :template: function.rst

   clustered_inference
   clustered_inference_pvalue
   desparsified_lasso
   desparsified_lasso_pvalue
   desparsified_group_lasso_pvalue
   ensemble_clustered_inference
   ensemble_clustered_inference_pvalue
   model_x_knockoff

Samplers
=======

.. autosummary::
   :toctree: ./generated/api/class/
   :template: class.rst

   conditional_sampling.ConditionalSampler

Helper Functions
================

.. autosummary::
   :toctree: ./generated/api/helper_functions/
   :template: function.rst

   quantile_aggregation
   reid
