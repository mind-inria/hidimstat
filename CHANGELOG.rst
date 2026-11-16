================================
under development (0.4.1.dev)
================================

..
    Changelog entry format:

    - :bdg-<color>:`Category` Description (:gh:`PR_NUMBER` by `Author`_).

    Categories:
      :bdg-success:`Feature`     new functionality
      :bdg-danger:`Fix`          bug fixes
      :bdg-primary:`Doc`         documentation, examples
      :bdg-warning:`API`         API changes, deprecations
      :bdg-secondary:`Maint`     CI, testing, maintenance, dependencies

Changes
-------

- :bdg-success:`Feature` Add the holdout randomization test (HRT). As a side effect, ``nadeau_bengio_ttest`` loses its ``axis`` argument and now always reduces over the last axis (:gh:`754` by `Joseph Paillard`_).
- :bdg-danger:`Fix` Removed unnecessary warning in ``fit_importance`` when (X, y) are passed to estimators that don't need it to compute importance scores (:gh:`804` by `Marc Hulcelle`_).
- :bdg-warning:`API` Deprecated feature importance functions such as ``cfi_importance`` to be removed in v0.6 (:gh:`783` by `Marc Hulcelle`_).
- :bdg-warning:`API` Removed deprecated classes ``CluDL`` and ``EnCluDL`` (:gh:`807` by `Marc Hulcelle`_).
- :bdg-warning:`API` Deprecated ``method`` and ``loss`` parameters of classes to replace with sklearn compatible ``scoring`` (:gh:`787` by `Marc Hulcelle`_).
- :bdg-warning:`API` Re-orded class inheritance with ``GroupVariableImportanceMixin`` (:gh:`788` by `Marc Hulcelle`_).
- :bdg-primary:`Doc` Added an example giving an overview of a feature importance analysis with ``hidimstat`` (:gh:`800` by `Joseph Paillard`_).
- :bdg-primary:`API` Added sklearn-compatible feature selectors (SelectTopK / PValueSelect / FDRSelect / FWERSelect) (:gh:`824` by `Marc Hulcelle`_).
- :bdg-warning:`API` Moved parameter checks to ``fit()`` and ``importance()`` methods (:gh:`827` by `Marc Hulcelle`_).
- :bdg-warning:`API` Swapped ModelXKnockoffs importance dimensions for API consistencies (:gh:`825` by `Marc Hulcelle`_).
- :bdg-warning:`API` Exposing fitted conditional samplers and adding a ``feature_predictability_`` attribute in CFI (:gh:`829` by `Marc Hulcelle`_).
- :bdg-warning:`API` Deleted custom ``_check_fit()`` method for more sklearn compliance (:gh:`834` by `Marc Hulcelle`_).
- :bdg-secondary:`Maint` Added upper bound to pandas version due to sklearn incompatibility (:gh:`840` by `Marc Hulcelle`_).
- :bdg-primary:`Doc` Added an onboarding document for new maintainers (:gh:`838` by `Marc Hulcelle`_).
