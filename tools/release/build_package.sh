#!/usr/bin/bash

rm -f -r build/*
rm -f -r dist/*

python -m pip install --upgrade pip
pip install --upgrade setuptools wheel twine

python setup.py sdist bdist_wheel
twine upload dist/*
