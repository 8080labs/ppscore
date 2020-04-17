#!/bin/bash

if [ -d ./dist ]
then
    rm -rf dist/*
fi

python3 setup.py sdist  # use miniconda3/bin/python3
twine upload dist/*