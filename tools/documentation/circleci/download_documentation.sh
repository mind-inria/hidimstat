#!/bin/bash

set -x -e

wget $GITHUB_ARTIFACT_URL
mkdir -p docs/_build/html
unzip doc*.zip -d docs/_build/html
