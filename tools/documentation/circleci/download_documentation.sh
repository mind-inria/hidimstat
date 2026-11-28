#!/bin/bash

set -x -e

# nightly.link can briefly return server errors
# right after the GitHub workflow that produced the artifact finished,
# so retry a few times before giving up.
wget \
    --tries=5 \
    --waitretry=30 \
    --retry-on-http-error=500,502,503,504 \
    --output-document=doc.zip \
    "$GITHUB_ARTIFACT_URL"
mkdir -p docs/_build/html
unzip doc.zip -d docs/_build/html
