===================================
Version 0.4.0 (under devellopement)
===================================

HIGHLIGHTS
----------

New features
------------

* documentation: added high dimension section (PR #425)
* API: ``BasePerturbation`` can now be initialized with unfitted estimators. When this the case, the estimator will first be fitted before variable importance is estimated. (PR #571)

Changes
-------


Bug fixes
---------

* Fixed a bug in CLuDL/EncluDL: cloning the internal estimators to avoid them carrying unwanted history. (PR #425)

Maintenance
-----------

Contributors
------------

* Bertrand Thirion
* Joseph Paillard
* Rémi Gau
  
