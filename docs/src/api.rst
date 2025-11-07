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

   clustered_inference
   clustered_inference_pvalue
   ensemble_clustered_inference
   ensemble_clustered_inference_pvalue

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

   ~statistical_tools.ConditionalSampler
   ~statistical_tools.GaussianKnockoffs

Helper Functions
================

.. autosummary::
   :toctree: ./generated/api/helper_functions/
   :template: function.rst

   ~statistical_tools.aggregation.quantile_aggregation
   ~desparsified_lasso.reid
   ~statistical_tools.nadeau_bengio_ttest
