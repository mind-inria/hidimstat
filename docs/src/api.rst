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

   ~base_variable_importance.BaseVariableImportance
   ~base_perturbation.BasePerturbation
   ~base_variable_importance.GroupVariableImportanceMixin

Feature Importance Classes
==========================

.. autosummary::
   :toctree: ./generated/api/class/
   :template: class.rst
   
   LOCO
   CFI
   PFI
   D0CRT
   ModelXKnockoff
   DesparsifiedLasso

Feature Importance functions
============================

.. autosummary::
   :toctree: ./generated/api/class/
   :template: function.rst
   
   cfi_importance
   clustered_inference
   clustered_inference_pvalue
   d0crt_importance
   desparsified_lasso_importance
   model_x_knockoff_importance
   ensemble_clustered_inference
   ensemble_clustered_inference_pvalue
   loco_importance
   pfi_importance

Visualization
=============

.. autosummary::
   :toctree: ./generated/api/class/
   :template: class.rst

   ~visualization.PDP


Samplers
========

.. autosummary::
   :toctree: ./generated/api/class/
   :template: class.rst

   ~samplers.ConditionalSampler
   ~samplers.GaussianKnockoffs

Helper Functions
================

.. autosummary::
   :toctree: ./generated/api/helper_functions/
   :template: function.rst

   ~statistical_tools.aggregation.quantile_aggregation
   ~desparsified_lasso.reid
   ~statistical_tools.nadeau_bengio_ttest
