Continuous Integration (CI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
After creating your PR, CI tools will run proceed to run all the tests on all
configurations supported by hidimstat.

- **Github Actions**:
  Used for testing hidimstat across various platforms (Linux, macOS, Windows).
- **CircleCI**:
  Builds and verifies the project documentation.

If any of the following markers appears in the commit message, the following
actions are taken.

    ====================== ===================
    Commit Message Marker  Action Taken by CI
    ---------------------- -------------------
    [skip tests]           The tests are not run
    ====================== ===================

There is a dependence between the differents part of the CI. You need to pass the
static analisis by the linter (Black) before that the example are running.
The documentation is build onlu if the tests are succeed.

For more details, the Github Action create an event on CircleCI pipeline which
 triger the building of the documentation and the deployment of the documentation.