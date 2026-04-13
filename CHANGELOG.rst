# under devellopement (0.3.0.dev)

## HIGHLIGHTS

## New features

* documentation: added high dimension section (PR #425)
* API: ``BasePerturbation`` can now be initialized with unfitted estimators. When this the case, the estimator will first be fitted before variable importance is estimated. (PR #571)
* Doc: add D0CRT section in the user-guide (PR #621)
* Doc: add example with MEG somato data (PR #607)

## Changes

## Bug fixes

* Fixed a bug in CLuDL/EncluDL: cloning the internal estimators to avoid them carrying unwanted history. (PR #425)
* Fix sign in D0CRT-logit (PR #627)
* Test: relax flaky test for conditional sanmpling (PR #602)

## Maintenance

* Update pandas intersphinx mapping (PR #615)
* scikit-learn compatibility: run check_estimator on PFI and PFICV (PR #610)
* scikit-learn compatibility: run check_estimator on ModelXKnockoff (PR #608)
* CI: update python version (PR #606)
* Fix inconsistent attribute names in ModelXKnockoff (PR #600)
* update pre-commit hooks (PR #597)
* silence or solve unused arguments errors (PR #596)
* Fix docstring in scenario.py (PR #594)
* Test: set random generator as a test fixture (PR #593)
* Fix pandas and perf errors (PR #592)
* Fix ruff bugbear issues (PR #591)
* scikit-learn compatibility: run check_estimator on DesparsifiedLasso (PR #590)
* Fix simplify ruff errors (PR #589)
* Doc: silence some warnings (PR #588)
* Doc: fix CI bug (PR #586)

## Contributors

* Bertrand Thirion
* Joseph Paillard
* Rémi Gau
