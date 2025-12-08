---
title: hidimstat-release

---
# How to create a release

The simplest way to make a release can be find in this `tutorial <https://packaging.python.org/en/latest/tutorials/packaging-projects/>`

The creation of the release is based on a builder.
We use `setuptools` has builder. This following links, we helps to configure it:  
https://learn.scientific-python.org/development/guides/packaging-simple/
https://setuptools.pypa.io/en/latest/userguide/quickstart.html
https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html

The version of the packages is defined dynamically based on the git tag using setuptools_scm.
For more details about it, look at this pages:
https://setuptools-scm.readthedocs.io/en/stable/usage/#builtin-mechanisms-for-obtaining-version-numbers

The format of version tag is X.Y.Z:
  - X represents a major revision (frequency: more than 1 year):
    - Important modification of the API
    - Refactoring of the major part of the code
  - Y represents major release (frequency: 6 month to 1 year):
    - Add new functionality (methods, functionalities, ...)
    - Important modification of one function
  - Z represents minor release (minimum: 1 days):
    - Fix important bugs in a release
    - Small modification of the example
    - Adding new examples

## Step for creating the Release:
This steps suppose that there a branch for each a major release (0.0.X).
The main branch is used for the ongoing work on the next major release.
For a major revision, a new branch should be created from scratch or from a commit on main.
For a modification of a previous release, the modification should be pushed on the branch associated with it.
      # TODO check if it's necessary to have a script release/build_packages.py

0. [Check if today is a good day for releasing](https://shouldideploy.today/)

1\. Update the information related to the release:
  - Update `CHANGELOG.rst` with the missing elements
  - Update `CONTRIBUTORS.rst` with the missing contributors

2\. Update the docstring of function based on CHANGELOG with ``deprecated``, ``versionchanged`` and ``versionadded`` directives.
Add `        .. versionadded:: x.y.z` in the docstring.
Additionally, make sure all deprecations that are supposed to be removed with this new version have been addressed.

2\. Create create a tag and branches associate with the release.
   
   1. git commit --allow-empty-message -m 'release X.Y.Z'
   - **minor release of the ongoing release**:
     1. Create a branch `git checkout -b release-X.Y.(Z+1)`
     2. Add a tag on the last commit with the name of the release `git tag X.Y.(Z+1)`
   - **major release**:
     1. A create a branch like `git checkout -b X.(Y+1).0` on the last commit of main
     2. (optional) `git rebase -i X.Y.Z` if you want to cherry pick some commits  
     3. Add a tag to this branch with `git tag X.(Y+1).0`
   - **major revision**:
     1. `git reset --hard (X+1).0.0` # Change main to major revion branch (with a force push)
     2. `git push -f mind-inria/main` # force to update main (disable the rule Prevent Branch deletion)
     3. `git checkout (X+1).0.0` # switch to the branch
     4. Add a tag on this version `(X+1).0.0` (the branch should be already create) `git tag (X+1).0.0`

3\. build the wheel & test it
  - `cd $(root of repository)` 
  - `rm -r release_file`  # remove the previous build
  - `mkdir release_file`
  - `git pull` # update your repository
  - `git checkout X.Y.Z` # checkout to the tag
  - `python -m venv_release release_file/venv` # create virtual environment for testing the release
  - `source release_file/venv/bin/activate` # activate this new venvironement
  - `pip install build twine uv` # install packages for create a wheel and push it on pype
  - `python -m build -o release_file/dist` (may need `pip install build`) # build the wheel
  - `twine check release_file/dist/*` (may need `pip install twine`) # check if the wheel is ready to be use
  - `pip install release_file/dist/hidimstat.....whl` # install the wheel in a fresh virtualenv
  - `uv pip install -r pyproject.toml --extra test; pytest` # test the installation
  
4\. Create a PR for creating the release:
   1. `git push origin tag X.Y.Z`
   2. Create a PR based on this new branch to the **right branch**.
   
Merging this PR will update the documentation automatically

5a. If you are on main => Update the changelog, contributor and version:
   - Move the `CHANGELOG.rst` in the `docs/src/whats_news/vX.Y.Z.rst`
      `mv CHANGELOG.rst docs/whats_news/vX.Y.Z.rst`
   - Update symlink to latest version of the changelog: `rm docs/src/whats_news/latest.rst`
   `cd docs/src/whats_news/; ln -s ./vX.Y.Z.rst ./latest.rst`
   - Replace `CHANGELOG.rst` with an empty template of `build_tools\template\CHANGELOG.rst`
      `cp build_tools\template\CHANGELOG.rst CHANGELOG.rst`
   - Change the version in this template and commit the modification
   - Create a new entry in `doc_conf\whats_news\whats_news.rst`
   - Update the file `CONTRIBUTORS.rst` in the documentation
      `cp CONTRIBUTORS.rst doc_conf\whats_news\CONTRIBUTORS.rst`
   - Add/Update the documentation with the new version. For doing it's require to update `doc_conf/version.json` which define all the version of the project.\
   The 2 first elements indicate the devellopement version and the last stable version.
  - Update symlink to stable version in the github branches repo: https://github.com/hidimstat/hidimstat.github.io
  

5b. For minor release => Update the changelog, contributor:
  - Modify the `build_tools\template\CHANGELOG.rst` with the modification
  - Update `CONTRIBUTORS.rst` if it's necessary

6\. Commit and push modification:
   - Commit the modifications
   - Push the modification 

7\. merge the PR on `X.Y.Z` (don't squash the commits)
  - check if the tests are passed and the redering of the documentation, the examples, the changelog are good
  - merge the PR **without squashing commit**:  
  no squash see warning in https://scikit-learn.org/dev/developers/maintainer.html#reference-steps \
  *NOTE*: in normal times only squash&merge is enabled because that's what we want for the main branch and we don't want to rebase or merge without squashing my mistake. there seems to be no way to configure this per branch ATM on github. so when we do a release we temporarily enable rebase. go to repository settings -> general -> pull requests, enable rebase, then merge the PR on 0.4.X (with the rebase option), then in the settings disable it again
- now we build the wheel we will upload to pypi locally `git fetch upstream` and `git checkout upstream/X.Y.Z`

7\. Rebuild the wheel & retest it (see step [3])

8\. (Optional) upload to TestPyPi for testing (https://test.pypi.org/)
  - `twine upload --repository testpypi release_file/dist/*`
  - `python3 -m pip install --upgrade --force-reinstall ---index-url https://test.pypi.org/simple/ --no-deps --extra-index-url https://test.pypi.org/simple/hidismtat`
  - `pytest` # test the installation

9\. upload to pype
  - `twine upload release_file/dist/*`
  - (Optional) `python3 -m pip install --upgrade --force-reinstall --no-deps hidismtat==X.Y.Z`
  - (Optional) `pytest` # test the installation

10\. Update the tag
  - Update the tag: `git tag -d X.Y.0`
  - `git tag -s 'X.Y.Z'` # `-s` is for signing, optional
  - `git push upstream X.Y.Z` # (disable the rule Prevent Branch deletion)

11\. Create a release on github from a specific tag:
  - See the following tutorial: https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository#about-release-management
  At this point, we need to upload the binaries to GitHub and link them to the tag.
  To do so, go to the :nilearn-gh:`Hidimstat GitHub page <tags>` under the "Releases" tab, and edit the ``x.y.z`` tag by providing a description, and upload the distributions we just created (you can just drag and drop the files).

12\. Update the conda-forge recipe
  - in `hidimstat-feedstock` https://github.com/conda-forge/hidimstat-feedstock
  - create branch `release-X.Y.Z`
  - update `recipe/meta.yml`
    - that is the only file we ever edit manually in that repo
    - update version number
    - update sha256
    - if needed reset build number to 0
    - if needed update the requirements (easiest way to check is in hidimstat `git checkout X.Y.Z` `git diff X.Y.(Z-1) -- pyproject.toml`)
  - open a PR to `upstream/hidimstat-feedstock` main branch
    - use checklist that will be posted in PR
    - in particular it asks to post a comment asking a bot to re-render the
      recipe, make sure to wait until that has finished
    - then merge the PR. it can take around an hour (maybe more?) for the
      package to be available from the conda-forge channel
    - when it becomes available, install in a fresh env & test
    - NOTE: to add new maintainers to that repo add them to the list at the end of meta.yml

13\. Once everything is done take a break by announced the release on social network channels.
