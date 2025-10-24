#!/bin/bash
# This script is meant to be called in the "deploy" step defined in
# circle.yml. See https://circleci.com/docs/ for more details.
# The behavior of the script is controlled by environment variable defined
# in the circle.yml in the top level folder of the project.

set -e
set -x

GENERATED_DOC_DIR=$1

if [[ -z "$GENERATED_DOC_DIR" ]]; then
    echo "Need to pass directory of the generated doc as argument"
    echo "Usage: $0 <generated_doc_dir>"
    exit 1
fi

# Absolute path needed because we use cd further down in this script
GENERATED_DOC_DIR=$(readlink -f $GENERATED_DOC_DIR)

if [ "$CIRCLE_BRANCH" = "main" ]
then
    dir=dev
else
    # Strip off .X
    dir="${CIRCLE_BRANCH}"
fi

MSG="Pushing the docs to $dir/ for branch: $CIRCLE_BRANCH, commit $CIRCLE_SHA1"

cd $HOME
if [ ! -d $DOC_REPO ];
then git clone "git@github.com:"$ORGANIZATION"/"$DOC_REPO".git";
fi
cd $DOC_REPO

# reset the documentation repository
git checkout main
git reset --hard origin/main
# delete all the documentation
git rm -rf $dir/ && rm -rf $dir/
# copy the new documentation
cp -R $GENERATED_DOC_DIR $dir
# commit the change
git config --global user.email $EMAIL
git config --global user.name $USERNAME
git config --global push.default matching
git add -f $dir/
git commit -m "$MSG" $dir
# push the modification
git push

echo $MSG
