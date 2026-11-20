.. _how_to_release:

===============================
How to create a release ?
===============================

This section explains how to create a release for hidimstat. This guide is destined
to developers with the adequate rights.

.. contents:: Table of Contents
   :depth: 2
   :local:

Introduction
------------

The creation of the release is based on a specific tool called a builder, ``setuptools``.
This tool normally doesn't have to be called directly here, as all the configuration
has already been written in the ``pyproject.toml`` file.

The format of version tag is X.Y.Z:

- X represents a major revision (frequency: more than 1 year):

  - Important modification of the API
  - Refactoring of the major part of the code

- Y represents major release (frequency: 6 months to 1 year):

  - Add new functionality (methods, functionalities, ...)
  - Important modification of one function

- Z represents minor release (minimum: 1 day):

  - Fix important bugs in a release
  - Small modification of the example
  - Adding new examples

Some useful links for tool configuration or to write this guide were saved :ref:`here <release_useful_links>`.
Please have a quick look if you want to, or face an issue during release.
This guide should be enough to walk you through the different steps.

Release steps
-------------

The release needs to be executed from a branch that follows a specific naming convention,
due to how to CI is built: ``release_X.Y.Z``
This branch is created from the main branch, and the following steps should be executed
only once all target modifications for the new version have been included on the release branch.

0. `Check if today is a good day for releasing <https://shouldideploy.today/>`_

1. Update the information related to the release:

   - Update ``hidimstat\CHANGELOG.rst`` with the missing elements.
   - Update ``hidimstat\CONTRIBUTORS.rst`` with the missing contributors for the new version.

2. Update the docstring of function based on CHANGELOG with ``deprecated``, ``versionchanged``
   and ``versionadded`` directives in the docstrings under a ``Note`` section.
   Additionally, make sure all deprecations that are supposed to be removed with this new version
   have been removed.

3. Update the changelog, contributor and version:

   - Move the ``CHANGELOG.rst`` in the directory ``docs/src/whats_news/``
     and rename it according to the target version `vX.Y.Z.rst`
     .. code::

        mv CHANGELOG.rst docs/whats_news/vX.Y.Z.rst

   - Update symlink to latest version of the changelog:
     .. code::

        rm docs/src/whats_news/latest.rst``
        cd docs/src/whats_news/; ln -s ./vX.Y.Z.rst ./latest.rst

   - Replace ``CHANGELOG.rst`` with an empty template of ``hidimstat\docs\tools\_templates\CHANGELOG.rst``
     .. code::

        cp docs\tools\_templates\CHANGELOG.rst .\CHANGELOG.rst

   - Change the version in this template and commit the modification
   - Create a new entry in ``docs\src\whats_news\whats_news.rst``
   - Update the file ``CONTRIBUTORS.rst`` in the documentation
     .. code::

        cp CONTRIBUTORS.rst docs\src\whats_news\CONTRIBUTORS.rst

   - Update the file ``docs\tools\version.json`` which defines all the versions of the project.
     The first two elements indicate the development version and the last stable version.
     Write a new entry for the current version, update the last stable version, and the new dev version.
   - Update symlink to stable version in the `github branches repo <https://github.com/hidimstat/hidimstat.github.io>`_
     This update requires special authorization, so ask admins/main maintainers for edition authorization.

4. Create a tag and branches associate with the release.
   Please be aware that once you create the tag, the tag head will be detached from the release branch.
   If you have to make changes after this, update the tag as described in Step 8.
   .. code::

    git commit --allow-empty-message -m 'release X.Y.Z'
    # minor release of the ongoing:
    git checkout -b release_X.Y.(Z+1) # Create a branch
    git tag X.Y.(Z+1) # Add a tag on the last commit with the name of the release
    #major release:
    git checkout -b release_X.(Y+1).0 # Create a branch on the last commit of main
    git rebase -i release_X.Y.Z # (optional) if you want to cherry pick some commits
    git tag X.(Y+1).0 # Add a tag to this branch
    #major revision
    git reset --hard (X+1).0.0` # Change main to major revion branch (with a force push)
    git push -f mind-inria/main` # force to update main (disable the rule Prevent Branch deletion)
    git checkout release_(X+1).0.0` # switch to the branch
    git tag (X+1).0.0 # Add a tag on this version `(X+1).0.0`

5. Build the wheel & test it
   .. code::

    cd $(root of repository)
    rm -r release_file  # remove the previous build
    mkdir release_file
    git pull # update your repository
    git checkout X.Y.Z # checkout to the tag
    python -m venv release_file/venv_release # create virtual environment for testing the release
    source release_file/venv_release/bin/activate # activate this new venvironement
    pip install build twine uv # install packages for create a wheel and push it on pype
    python -m build -o release_file/dist # build the wheel
    twine check release_file/dist/* # check if the wheel is ready to be use
    pip install release_file/dist/hidimstat.....whl # install the wheel in a fresh virtualenv
    uv pip install -r pyproject.toml --extra test; pytest # test the installation

6. Push the tag, and push and create a PR for the release branch, if not done yet:
   .. code::

    git push origin tag X.Y.Z`
    git checkout release_X.Y.Z`  # return to the release branch
    git push origin`  # push the release branch
    # Then create the PR for the release branch, with the main branch as target.

7. Commit and push any modifications

8. Update the tag if any modifications were pushed at the previous step
   .. code::

    git tag -d X.Y.Z # Delete the tag. This requires an exception on the tag deletion rule of the repository (Settings/Rulesets/Prevent Tag Deletion)
    git tag -s X.Y.Z # `-s` is for signing, optional
    git push origin X.Y.Z # (disable the rule Prevent Branch deletion)

9. merge the PR on ``release_X.Y.Z`` (don't squash the commits)

   - check if the tests pass, the rendering of the documentation, the examples and the changelog are good
   - merge the PR **without squashing commit**:
     Normally, only squash & merge is enabled. There seems to be no way to configure this per branch ATM on github.
     When we do a release, we temporarily enable rebase. To do so, go to repository settings -> general -> pull requests,
     enable rebase, then merge the PR on release_X.Y.Z (with the rebase option).
     Once done, switch to squash and merge in the settings again.
     Once the PR is merged, the documentation will automatically be updated by the CI.

10. Rebuild the wheel & retest it (see step [3]):
    .. code::

        git fetch origin
        git checkout origin/X.Y.Z
        # Follow step 3 instructions

11. (Optional) Upload to TestPyPi for testing (`https://test.pypi.org/ <https://test.pypi.org/>`_)
    This requires an authorization for the organization, and the creation of an access token.
    .. code::

        twine upload --repository testpypi release_file/dist/*
        python3 -m pip install --upgrade --force-reinstall --index-url https://test.pypi.org/simple/ --no-deps --extra-index-url https://test.pypi.org/simple/hidismtat
        pytest # test the installation

12. Upload to `PyPi <https://pypi.org/>`_. This requires a separate authorization from TestPyPi for the organization,
    and the creation of an access token that is different from TestPyPi.
    .. code::

        twine upload release_file/dist/*
        python3 -m pip install --upgrade --force-reinstall --no-deps hidimstat==X.Y.Z
        pytest # test the installation

13. Create a release on github from a specific tag:
    At this point, we need to upload the binaries (what we have just built) to GitHub and link them to the tag.
    To do so, go there: `https://github.com/mind-inria/hidimstat/tags <https://github.com/mind-inria/hidimstat/tags>`_
    and edit the tag by providing a description (copy and paste the content of the ``docs\src\whats_news\X.Y.Z.rst``),
    and upload both build files that we created situated in the folder ``release_file\dist``.

14. Update the conda-forge recipe

    - in `hidimstat-feedstock <https://github.com/conda-forge/hidimstat-feedstock>`_
    - create branch ``release_X.Y.Z``
    - update ``recipe/meta.yml``

      * that is the only file we ever edit manually in that repo
      * update version number
      * update sha256
      * if needed reset build number to 0
      * if needed update the requirements (easiest way to check is in hidimstat

        git checkout X.Y.Z
        git diff X.Y.(Z-1) -- pyproject.toml)

    - open a PR to ``mind-inria/hidimstat-feedstock`` main branch

      * use checklist that will be posted in PR
      * in particular it asks to post a comment asking a bot to re-render the
        recipe, make sure to wait until that has finished
      * then merge the PR. it can take around an hour (maybe more?) for the
        package to be available from the conda-forge channel
      * when it becomes available, install in a fresh env & test
      * NOTE: to add new maintainers to that repo add them to the list at the end of meta.yml

15. Congratulations, the release is over !
