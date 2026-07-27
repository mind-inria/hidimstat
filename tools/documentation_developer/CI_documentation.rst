.. _developer_documentation_CI:

Continuous Integration (CI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
After creating your PR, CI tools will proceed to run all the tests on all
configurations supported by hidimstat.

- **Github Actions**:
  Used for testing hidimstat across various platforms (Linux, macOS, Windows).
- **CircleCI**:
  Builds and verifies the project documentation.

If any of the following markers appear in the committed message, the following
actions are taken.

    ====================== ===================
    Commit Message Marker  Action Taken by CI
    ---------------------- -------------------
    [skip tests]           The tests are not run
    [doc skip]             Docs are not built
    [doc quick]            Docs built, but excludes example gallery plots
    [doc change]           Docs built, including only example, which has been directly modified
    ====================== ===================

There is a dependence on the different parts of the CI. You need to pass the
static analysis by the linter (Black) before that the example is running.
The documentation is built only if the tests succeeded.

Note that, by default, the documentation is built with all the examples are modified.

For more details, the GitHub Action creates an event on the CircleCI pipeline, which
triggers the documentation build and the deployment of the documentation.

CI is testing all possible configurations supported by hidimstat, so tests may fail
with configurations different from your development setup. See with the
maintainers to identify the cause of any possible failure.

Test results are posted on a CircleCI workspace that is accessible by clicking on the
associated Github actions at the bottom of a pull-request's page.


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
