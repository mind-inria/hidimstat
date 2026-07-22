.. _test_suite:

==============
The Test Suite
==============

.. contents:: On this page
   :local:
   :depth: 2

Overview
========

The test suite works under ``pytest``. The suite tests three different elements for every
inference method:

- a correctness/statistical-power test on simulated data
- an exceptions/error-handling test class, and, where the object wraps a scikit-learn estimator
- a scikit-learn estimator guidelines compliance test.

We present here the structure of the test suite organization, the logic behind tests, and quickly
present important fixtures, and other important utility functions.

Tests are automatically being run after each push on a pull-request (PR) whose target branch is ``main``. They are
run on all combinations of python versions in ``[3.10, 3.14]``, and different OS structures
``[Windows, Mac, Linux]``. We target a maximal code coverage with tests, meaning that tests should
cover every line of existing code. To ensure this, we have the ``CodeCov`` utility bot that
automatically computes the code coverage of tests after a successful commit on a PR, meaning that
the entire test suite run is successful, and that the documentation generation process by ``Sphinx``
is executed without any error. The code coverage metric should then appear after the execution of tests
and documentation generation process directly on the PR discussion on `GitHub <https://github.com/mind-inria/hidimstat/pulls>`_.

How the Suite Is Organized
==========================

File Inventory
--------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - File
     - What it tests
   * - ``conftest.py``
     - Shared pytest fixtures and helpers (no tests of its own)
   * - ``__init__.py``
     - Empty — makes the test folder a package so relative imports
       (``from .conftest import ...``) work
   * - ``test_version.py``
     - Sanity check that ``hidimstat.__version__`` is a valid string
   * - ``test_base_perturbation.py``
     - ``BasePerturbation`` / ``BasePerturbationCV`` base classes shared
       by perturbation-style importance methods
   * - ``test_base_variable_importance.py``
     - ``BaseVariableImportance`` — the shared selection/plotting logic
       (top-k, percentile, threshold, FDR, FWER selection; plotting)
   * - ``test_permutation_feature_importance.py``
     - ``PFI`` / ``PFICV`` (Permutation Feature Importance)
   * - ``test_conditional_feature_importance.py``
     - ``CFI`` / ``CFICV`` (Conditional Feature Importance)
   * - ``test_leave_one_covariate_in.py``
     - ``LOCI`` / ``LOCICV`` (Leave-One-Covariate-In)
   * - ``test_leave_one_covariate_out.py``
     - ``LOCO`` / ``LOCOCV`` (Leave-One-Covariate-Out)
   * - ``test_desparsified_lasso.py``
     - ``DesparsifiedLasso``, the ``reid``/group-Reid noise estimators
   * - ``test_knockoff.py``
     - ``ModelXKnockoff``, ``GaussianKnockoffs`` (Model-X knockoffs)
   * - ``test_distilled_conditional_randomization_test.py``
     - ``D0CRT`` (distilled conditional randomization test / dCRT)
   * - ``test_ensemble_clustered_inference.py``
     - ``CluDL`` / ``EnCluDL`` (clustered and ensembled-clustered
       desparsified Lasso inference on spatial/temporal data)
   * - ``_utils/``
     - The data generation methods, and various method specific
       and non-specific utility functions.
   * - ``samplers/``
     - Conditional sampling functions, and samplers for Gaussian knockoffs.
   * - ``statistical_tools/``
     - Statistical tests functions for test aggregations, multiple testing,
       Nadeau-Bengio T-tests, and computations of p-values.


The test suite follows the convention that there is a single test file for each single
source python file. It ensures that each estimator or family of estimators has its own test
module named after the corresponding source module. This keeps tests discoverable and
lets a contributor working on one method run just ``pytest test_<method>.py``.

Test organization logic
-----------------------

For feature importance methods, tests fall into three categories:

- Statistical correctness on simulated data:
   Tests generate synthetic data with a known support (important vs.
   non-important features) via data simulation functions
   found in ``hidimstat._utils.scenario``, run the method, and
   check that:

   - importance scores are higher on truly important features than on
     null features,
   - a selection procedure keeps the empirical false discovery
     proportion (FDP) below the target level and achieves reasonable
     statistical power.

   Because these are stochastic tests, many are repeated over multiple
   seeds and the assertions compare an average FDP/power against a target plus a small tolerance, rather
   than requiring every single draw to succeed.

- API and exception behavior:
   Dedicated classes or functions check that
   invalid inputs raise the right error type with the expected message,
   that calling ``.importance()`` before ``.fit()`` fails clearly, and that
   warnings are raised as expected.

- Scikit-learn estimator compliance:
   As feature importance methods are written in compliance to scikit-learn standards,
   each one of them is checked against scikit-learn's own estimator conformance checks,
   so they behave like any other scikit-learn estimator.

Reproducibility testing
-----------------------

Modules that can be seeded run the same four-way check on ``random_state`` handling:

- an integer seed gives identical results across repeated calls and
  across separate instances;
- ``random_state=None`` gives different results on every call and
  every new instance;
- a NumPy ``Generator`` object is consumed statefully. Repeated calls on the same
  instance differ, but re-creating the generator with the same seed reproduces the original
  result.

This is important for any Monte Carlo–style method where reproducibility guarantees matter.

All methods that involve randomness must include a dedicated test suite. The following
test patterns help to ensure consistent behavior:

- ``test_<method>_repeatability``
- ``test_<method>_randomness_with_none``
- ``test_<method>_reproducibility_with_integer``
- ``test_<method>_reproducibility_with_rng``

Important fixtures and utilities
================================

Defined in the testing configuration file
-----------------------------------------

``pytest_configure(config)``
   Forces matplotlib's non-interactive ``Agg`` backend so plotting
   tests never pop up a window during a test run.

``rng`` *(function-scoped)*
   Returns a freshly seeded ``numpy.random.default_rng(42)``. Used
   throughout the suite whenever a test needs its own controlled
   randomness (shuffling arrays, drawing p-values, generating toy
   datasets) independent of the main data-generation fixture below.

``data_generator`` *(function-scoped, parametrized)*
   The workhorse fixture of the whole suite. It depends on eight
   parameters — ``n_samples``, ``n_features``, ``support_size``,
   ``rho``, ``seed``, ``value``, ``signal_noise_ratio``,
   ``rho_serial`` — normally supplied via
   ``@pytest.mark.parametrize`` on the test (or test class), and calls
   ``multivariate_simulation`` to build a linear-model dataset with a
   known sparse support. It returns
   ``(X, y, important_features, not_important_features)`` — the design
   matrix, the response, and the index arrays a test needs to check
   that a method ranks/selects the right features. Because the
   parametrize decorator sits on the *test*, different tests can reuse
   this one fixture with completely different problem sizes and noise
   regimes (e.g. high-dimensional ``n_features=200`` cases, correlated
   features via ``rho``, or noisy targets via ``signal_noise_ratio``).

``fitted_linear_regression()`` / ``_fitted_linear_regression()``
   A plain helper (not a fixture) that fits a ``LinearRegression`` on a
   tiny random 2-column dataset, used to seed estimator-check lists
   with an *already-fitted* model — several checks specifically probe
   fitted-vs-unfitted behavior.

``check_estimator(estimators, return_expected_failed_checks, valid=True)``
   The sklearn < 1.6 compatibility helper described above; not a
   fixture in the pytest sense, but a generator consumed by
   ``@pytest.mark.parametrize`` to build ``(estimator, check, name)``
   tuples for valid or intentionally-invalid checks.

Local file-specific fixtures
----------------------------

Several test files define their own narrower fixtures on top of the
shared ones, generally to avoid refitting an expensive model in every
test. Please make sure that you have checked module-scoped fixtures
before implementing a utility function. Fixtures can generally be found
at the beginning of the file.

Non-fixtures testing utilities
------------------------------

If you ever need a non-fixture utility function, please make sure that you check existing functions in the separate
``_utils``, ``samplers``, and ``statistical_tools`` modules before reimplementing a method.

Summary
=======

The suite is thought to ensure having reusable testing
recipes,  applied consistently to every inference method in the library.
The testing configuration file ``conftest.py`` supplies the two
fixtures ``rng`` and ``data_generator``, and one helper
``check_estimator`` that make this consistency possible, while each
test module adds a thin, method-specific layer of fixtures and
parametrizations on top.
