#!/usr/bin/env bash

# display all the command and associet argument when they are executed
set -x
# exit immediately if a command exits with a non-zero status.
set -e

# create an virtual environement and install dependece with uv
python -m venv .venv
source .venv/bin/activate
pip install uv
uv pip install -e ".[doc]"