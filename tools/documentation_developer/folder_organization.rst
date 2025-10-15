.. _folder_organization:

===================
Folder Organization
===================

📦hidimstat
 ┣ 📜.gitignore                 (file to ignore in the git)
 ┣ 📜CONTRIBUTING.rst           (page guide for your project's contributors)
 ┣ 📜LICENSE                    (license detail)
 ┣ 📜README.rst                 (page for introducing and explaining the project)
 ┣ 📜codecov.yml                (configuration file for CodeCov)
 ┗ 📜pyproject.toml             (configuration file for Python project)
 ┣ 📂.circleci                  (folder for CircleCI)
 ┃ ┗ 📜config.yml               (configuration file for CircleCI)
 ┣ 📂.github                    (folder for Github)
 ┃ ┣ 📂workflows                (folder for define the Github action)
 ┃ ┃ ┣ 📜call_linter.yml        (workflow for running the linter)
 ┃ ┃ ┣ 📜call_publish_result.yml(workflow for sending the results to CodeCov and CircleCI)
 ┃ ┃ ┣ 📜call_test_minimal.yml  (workflow for running test with minimal version)
 ┃ ┃ ┣ 📜call_test_package.yml  (workflow for running tests)
 ┃ ┃ ┣ 📜ci.yml                 (Action for run all the workflow from main)
 ┃ ┃ ┣ 📜ci_test.yml            (Action for testing a modification of CI)
 ┃ ┃ ┗ 📜circleci-artifact.yml  (Action for getting the link to CircleCI)
 ┃ ┗ 📜dependabot.yml           (configuration of dependabot)
 ┣ 📂docs                       (folder for the documentation)
 ┃ ┣ 📂(T)_build                (folder which contains the generated documentation)
 ┃ ┃ ┣ 📂(T)doctrees            (contains the generated tree)
 ┃ ┃ ┗ 📂(T)html                (contains the generated HTML)
 ┃ ┣ 📂src                      (folder with the documentation page)
 ┃ ┃ ┣ 📂dev                    (redirection to tools/documentation_developer)
 ┃ ┃ ┣ 📂(T)generated           (contains the geneterated file for API and examples)
 ┃ ┃ ┣ 📜api.rst                (page for API)
 ┃ ┃ ┣ 📜index.rst              (page for the index)
 ┃ ┃ ┣ 📜user_guide.rst         (page of the user guide)
 ┃ ┃ ┗ 📜.....rst               (other pages of the documentation)
 ┃ ┣ 📂tools                    (folder for configuration of Sphinx)
 ┃ ┃ ┣ 📂_static                (folder for static part of html)
 ┃ ┃ ┃ ┣ 📜logo.png             (logo of HiDimStat)
 ┃ ┃ ┃ ┗ 📜style.css            (css for the library)
 ┃ ┃ ┣ 📂_templates             (Folder of template for autodoc Sphinx)
 ┃ ┃ ┃ ┣ 📜class.rst            (Template for classes)
 ┃ ┃ ┃ ┗ 📜function.rst         (Template for function)
 ┃ ┃ ┣ 📜conf.py                (configuration of Sphinx)
 ┃ ┃ ┣ 📜references.bib         (reference for the citation)
 ┃ ┃ ┗ 📜utils.py               (functions use in the configuration file)
 ┃ ┗ 📜Makefile                 (Makefile for generated the documentation)
 ┣ 📂examples                   (Folder contains all the examples)
 ┃ ┣ 📜README.txt               (index of examples for sphinx-gallery)
 ┃ ┗ 📜plot_....py ()           (examples)
 ┣ 📂src                        (folder containing the source code)
 ┃ ┗ 📂hidimstat                (code of the library)
 ┃ ┃ ┣ 📂_utils                 (folder for private functions)
 ┃ ┃ ┃ ┣ 📜__init__.py          
 ┃ ┃ ┃ ┣ 📜bootstrap.py         (module for bootstrap handling)
 ┃ ┃ ┃ ┣ 📜docstring.py         (function for docstring)
 ┃ ┃ ┃ ┣ 📜exception.py         (function for the exceptions)
 ┃ ┃ ┃ ┣ 📜regression.py        (function for Lasso regression)
 ┃ ┃ ┃ ┣ 📜scenario.py          (function for generating data)
 ┃ ┃ ┃ ┗ 📜utils.py             (support functions)
 ┃ ┃ ┣ 📂statistical_tools      (folder which contains the statistical function:
 ┃ ┃ ┃ ┣ 📜__init__.py          test, sampler, aggregation, fdr)
 ┃ ┃ ┃ ┣ 📜aggregation.py
 ┃ ┃ ┃ ┣ 📜multiple_testing.py
 ┃ ┃ ┃ ┣ 📜sampler.....py
 ┃ ┃ ┃ ┗ 📜p_values.py
 ┃ ┃ ┣ 📜__init__.py            (definition of functions and classes)
 ┃ ┃ ┣ 📜(T)_version.py         (version of the library)
 ┃ ┃ ┣ 📜base_....py            (Abstract and Mixin classes)
 ┃ ┃ ┗ 📜.........py            (Conditional feature importance)
 ┣ 📂test                       (folder for the tests)
 ┃ ┣ 📂_utils                   (test for function in utils)
 ┃ ┃ ┗ 📜test_.....py
 ┃ ┣ 📂baseline_plots           (image of references)
 ┃ ┃ ┗ 📜test_.......png
 ┃ ┣ 📂statistical_tools        (test for function in statistical_tools)
 ┃ ┃ ┗ 📜test_........py
 ┃ ┣ 📜__init__.py              
 ┃ ┣ 📜conftest.py              (configuration tests)
 ┃ ┗ 📜test_..........py
 ┣ 📂tools                      (folder for develloper tools)
 ┃ ┣ 📂documentation            (tools for generatring and debug documentation)
 ┃ ┃ ┣ 📂circleci               (script used by circleci for the documentation)
 ┃ ┃ ┃ ┣ 📜build_doc.sh         (create the documentation)
 ┃ ┃ ┃ ┣ 📜checkout_merge_commit.sh     (checkout to right commit)
 ┃ ┃ ┃ ┣ 📜push_doc.sh          (push the documentation in git repository)
 ┃ ┃ ┃ ┗ 📜setup_virtual_environment.sh (setup the environment for the generation)
 ┃ ┃ ┗ 📂debugger_script        (folder with script for helping debugging)
 ┃ ┃ ┃ ┗ 📜.....py
 ┃ ┣ 📂documentation_developer  (Documentation for develloper)
 ┃ ┃ ┣ 📜index.rst              (index of documentation)
 ┃ ┃ ┗ 📜.......rst
 ┃ ┗ 📂examples                 (Script for debugging the generation of examples)
 ┃ ┃ ┗ 📂debugger_script       
 ┃ ┃ ┃ ┗ 📜try_reproducibility.py
 