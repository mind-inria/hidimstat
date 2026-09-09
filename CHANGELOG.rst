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
