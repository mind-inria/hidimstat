.. _developer_documentation_CI:

Continuous Integration (CI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

After creating your PR, CI tools will proceed to run all the tests on all
configurations supported by hidimstat.

- **Github Actions**:
  Used for testing hidimstat across various platforms (Linux, macOS, Windows)
  and building the project documentation.
- **CircleCI**:
  Host the project documentation for pull requests.

If any of the following markers appear in the committed message, the following
actions are taken.

    ============================  =======================================
    Commit Message Marker         Action Taken by CI
    ---------------------- -----  ---------------------------------------
    [skip tests]                  the tests are not run
    [skip doc]                    skip build of the documentation
    [full doc]                    runs a full build on pull-request
    [example] name_of_example.py  trigger build of some specific examples
    ============================  =======================================

Note that, by default, the documentation is built with all the modified examples.

For more details, the GitHub Action builds the documentation
and stores it as an artefact on https://nightly.link.
It then creates an event on the CircleCI pipeline,
which downloads the artefact and deploys of the documentation.

CI is testing all possible configurations supported by hidimstat,
so tests may fail with configurations different from your development setup.
See with the maintainers to identify the cause of any possible failure.

Test results are posted on a CircleCI workspace that is accessible
by clicking on the associated Github actions at the bottom of a pull-request's page.


Modifying Continuous Integration (CI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, only the workflow on the main branch is used for running the CI. This is
done for security reasons.
In consequence, to test a modification of the actual workflow or a new workflow,
contributors should create a pull request which contains modifications on .github/workflow/ci_test.yml.
You need to add the label `testCI` to automatically trigger the test of the CI.
You can also trigger the workflow using with the
`HTTP POST request <https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#repository_dispatch>`_.

Once this modification is merged into main, it should be important to clean ci_test.yml
for having an empty workflow.
