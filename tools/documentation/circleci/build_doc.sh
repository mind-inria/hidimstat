#!/usr/bin/env bash

# display all the command and associet argument when they are executed
set -x
# exit immediately if a command exits with a non-zero status.
set -e

# activate the environment for the creation of the documentation
# see the file setup_virtual_environment
source .venv/bin/activate

# Decide what kind of documentation build to run, and run it.
#
# If the last commit message has a "[doc skip]" marker, do not build
# the doc. On the contrary if a "[doc build]" marker is found, build the doc
# instead of relying on the subsequent rules.
#
# We always build the documentation for jobs that are not related to a specific
# PR (e.g. a merge to master or a maintenance branch).
#
# If this is a PR, do a full build if there are some files in this PR that are
# under the "doc/" or "examples/" folders, otherwise perform a quick build.
#
# If the inspection of the current commit fails for any reason, the default
# behavior is to quick build the documentation.

# check that the workflow is done on a Pull Request
if [ $(echo $CIRCLE_BRANCH | cut -d'/' -f 1) == 'pull' ]
then CI_PULL_REQUEST="PR"
fi

get_build_type() {
    # Full build if it is not in a PR
    if [ -z "$CI_PULL_REQUEST" ]
    then
        echo BUILD: not a pull request
        return
    fi
    # get the hash of the last commit of the PR
    if [ -z "$CIRCLE_SHA1" ]
    then
        echo SKIP: undefined CIRCLE_SHA1
        return
    fi
    # get the log of commit and detect marker: [doc skip], [doc quick], [doc changed]
    commit_msg=$(git log --format=%B -n 1 $CIRCLE_SHA1)
    if [ -z "$commit_msg" ]
    then
        echo QUICK BUILD: failed to inspect commit $CIRCLE_SHA1
        return
    fi
    if [[ "$commit_msg" =~ \[doc\ skip\] ]]
    then
        echo SKIP: [doc skip] marker found
        return
    fi
    if [[ "$commit_msg" =~ \[doc\ quick\] ]]
    then
        echo QUICK: [doc quick] marker found
        return
    fi
    if [[ "$commit_msg" =~ \[doc\ change\] ]]
    then
        # get the example difference between main and the actual commit
        git_range="origin/main...$CIRCLE_SHA1"
        git fetch origin main >&2 || (echo QUICK BUILD: failed to get changed filenames for $git_range; return)
        filenames=$(git diff --name-only $git_range)
        # case where there is no modified file
        if [ -z "$filenames" ]
        then
            echo QUICK BUILD: no changed filenames for $git_range
            return
        fi
        # get the examples which have been modified
        changed_examples=$(echo "$filenames" | grep -e ^examples/)
        if [[ -n "$changed_examples" ]]
        then
            echo BUILD: detected examples/ filename modified in $git_range: $changed_examples
            pattern=$(echo "$changed_examples" | paste -sd '|')
            # pattern for examples to run is the last line of output
            echo "$pattern"
            return
        fi
        # case where there is no modified example
        echo QUICK BUILD: no examples/ filename modified in $git_range:
        echo "$filenames"
        return
    fi
    echo BUILD: build all the example by default
}

build_type=$(get_build_type)
# Skip examples
if [[ "$build_type" =~ ^SKIP ]]
then
    exit 0
fi
# generate all the example if it's push on main or on a previous version
if [[ "$CIRCLE_BRANCH" =~ ^main$|^release\_[0-9]+\.[0-9]+\.[0-9]$ && -z "$CI_PULL_REQUEST" ]]
then
    make_args="html"
elif [[ "$build_type" =~ ^QUICK ]]
# do not generate examples
then
    make_args="html-noplot"
elif [[ "$build_type" =~ ^'BUILD: detected examples' ]]
# generate only example which has been modified
then
    # pattern for examples to run is the last line of output
    pattern=$(echo "$build_type" | tail -n 1)
    make_args="html EXAMPLES_PATTERN=$pattern"
else
    make_args="html"
fi

# The pipefail is requested to propagate exit code
set -o pipefail && cd docs && make N_JOB=2 $make_args 2>&1 | tee ~/output_sphinx.txt
cd -

set +o pipefail


affected_doc_paths() {
    # generate a list of the file modified in the PR
    files=$(git diff --name-only origin/main...$CIRCLE_SHA1)
    # list of the modified documentation files
    echo "$files" | grep ^docs/src/.*\.rst | sed 's/^docs\/src\/\(.*\)\.rst$/\1.html/'
    # list of the modified examples
    echo "$files" | grep ^examples/.*.py | sed 's/^\(.*\)\.py$/generated\/gallery\/\1.html/'
    # list of the modified source file
    project_files=$(echo "$files" | grep 'src/hidimstat/')
    if [ -n "$project_files" ]
    then
        grep -hlR -f<(echo "$project_files" | sed 's/src\/hidimstat\//hidimstat\./') docs/_build/html/generated | cut -d/ -f4-
    fi
}

# generate a html page which list the modified files
if [ -n "$CI_PULL_REQUEST" ]
then
    echo "The following documentation files may have been changed by PR #$CI_PULL_REQUEST:"
    affected=$(affected_doc_paths)
    echo "$affected"
    (
    echo '<html><body><ul>'
    echo "$affected" | sed 's|.*|<li><a href="&">&</a></li>|'
    echo '</ul><p>General: <a href="index.html">Home</a> | <a href="api.html">API Reference</a> | <a href="auto_examples/index.html">Examples</a></p></body></html>'
    ) > 'docs/_build/html/_changed.html'
fi
