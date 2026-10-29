#!/bin/bash
#
# This script is used to check
# if the warnings thrown in the examples.
#
# It aims to help minimize the number of warnings thrown.
#

set -e
set -x

mkdir -p docs/tmp
grep --include "*.html" -rn "docs/_build/html/generated/gallery/examples" -e "Warning: " > docs/tmp/all_warnings.txt || true
grep --include "*.html" -rn "docs/_build/html/generated/gallery/examples" -e "DeprecationWarning: " > docs/tmp/deprecation_warnings.txt || true
grep --include "*.html" -rn "docs/_build/html/generated/gallery/examples" -e "FutureWarning: " > docs/tmp/future_warnings.txt || true
grep --include "*.html" -rn "docs/_build/html/generated/gallery/examples" -e "UserWarning: " > docs/tmp/user_warnings.txt || true
grep --include "*.html" -rn "docs/_build/html/generated/gallery/examples" -e "RuntimeWarning: " > docs/tmp/runtime_warnings.txt || true
