================================
under development (0.3.2.dev0)
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

- :bdg-secondary:`Maint` Update version in doc's dropdown menu to 0.3.1 (stable) and clean changelog (:gh:`637` by `Joseph Paillard`_).
- :bdg-danger:`Fix` Fix bug in the KO methods: ``threshold_mesh`` was computed but not assigned to the variable (:gh:`643` by `Bertrand Thirion`_ amd `Joseph Paillard`_).
- :bdg-secondary:`Maint` Add badges and functional links to the changelog (:gh:`644` by `Joseph Paillard`_).
- :bdg-secondary:`Maint` Remove extra term in variance of X-residuals (:gh:`649` by `Joseph Paillard`_).
- :bdg-primary:`Doc` Add documentation on how to contribute with issues, pull-requests, explanation of the CI process, and dev guidelines on class templates and folder architecture. (:gh:`653` by `Marc Hulcelle`_).
- :bdg-primary:`Doc` Add naming conventions for classes, files, and functions, as well as citation conventions (:gh:`647` and :gh:`648` by `Marc Hulcelle`_).
- :bdg-danger:`Fix` Fix typo in issue template (:gh:`659` by `Joseph Paillard`_).
- :bdg-primary:`Doc` Add an AGENTS.md file for AI agents, and AI disclosures to contribution guidelines (:gh:`655` by `Marc Hulcelle`_).
- :bdg-secondary:`Maint` Add a maintenance-related issue template (:gh:`666` by `Marc Hulcelle`_).
- :bdg-secondary:`Maint` Temporary fix for the CI upper-bounding scikit-learn to 1.9.0 (:gh:`669` by `Joseph Paillard`_).
- :bdg-danger:`Fix` Fix unnecessary copy operations of X when only a slice view is needed (:gh:`646` by `Marc Hulcelle`_).
- :bdg-secondary:`Maint` Remove pinned poosh dependency (problem with somato dataset solved) (:gh:`670` by `Joseph Paillard`_).
- :bdg-secondary:`Maint` Remove extra term in variance of X-residual (DOCRT). See  [Reid et al., A Study of Error Variance Estimation in Lasso Regression 2016](https://arxiv.org/pdf/1311.5274) for reference. (:gh:`649` by `Joseph Paillard`_).
- :bdg-success:`Feature` Add leave-one-covariate-in (LOCI) method (:gh:`679` by `Marc Hulcelle`_).
- :bdg-danger:`Fix` Fix deprecated n_alphas with sklearn LassoCV, as well as deprecated penalty for LogisticRegressionCV (:gh:`690` by `Marc Hulcelle`_).
- :bdg-secondary:`Maint` Fix conditional sampling test by verifying that sampler produces diverse samples. (:gh:`692` by `Joseph Paillard`_).
- :bdg-success:`Feature` Implement marginal SAGE (:gh:`674` by `Joseph Paillard`_).
- :bdg-success:`Feature` Parallelize importance in EnCluDL (:gh:`710` by `Marc Hulcelle`_).
- :bdg-success:`Feature` Implement generic ClusterImportance and EnsembleImportance (:gh:`714` by `Marc Hulcelle`_).
- :bdg-primary:`Doc` Fix shape inconsistencies in docstring for importance and selection arrays (:gh:`711` by `Marc Hulcelle`_).
- :bdg-primary:`Doc` Add documentation about test suite organization and general logic. (:gh:`712` by `Marc Hulcelle`_).
- :bdg-primary:`Doc` Add section on SAGE in the user guide (:gh:`736` by `Joseph Paillard`_).
- :bdg-primary:`Doc` Missing classes from API autosummary (:gh:`734` by `Marc Hulcelle`_).
- :bdg-success:`Feature` Implement Accumulated Local Effects (ALE) (:gh:`701` by `Aurélien Bazire`_).
- :bdg-success:`Feature` Add bootstrap for ALE confidence intervals (:gh:`703` by `Aurélien Bazire`_).
