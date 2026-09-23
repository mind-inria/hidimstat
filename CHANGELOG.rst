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
