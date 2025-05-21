.. _developer_documentation_CI:

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
    [doc skip]             Docs are not built
    [doc quick]            Docs built, but excludes example gallery plots
    [doc change]           Docs built including only example which has been directly modified
    ====================== ===================

Note that by default the documentation is built with all the example are modified.

CI is testing all possible configurations supported by hidimstat, so tests may fail
with configurations different from what you are developing with. See with the 
mainteners to identify the cause of this failure. 
