.. _developer_documentation_CI:

Continuous Integration (CI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
After creating your PR, CI tools will run proceed to run all the tests on all
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
The documentation is built only if the tests are succeeding.

Note that, by default, the documentation is built with all the examples are modified.

For more details, the GitHub Action creates an event on the CircleCI pipeline, which
 trigger the building of the documentation and the deployment of the documentation.

CI is testing all possible configurations supported by hidimstat, so tests may fail
with configurations different from what you are developing with. See with the 
maintainers to identify the cause of this failure. 

Modification Continuous Integration (CI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Actually, only the workflow on the main branch is used for running the CI. This is 
done for security reasons.
In consequence, for testing a modification of the actual workflow or a new workflow,
it requires creating a pull request, which is modified .github/workflow/ci_test.yml for
allowing the running of this modification.

Once this modification is merged into main, it should be important to clean ci_test.yml 
for having an empty workflow. 