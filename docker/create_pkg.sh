#!/bin/bash

PROJ_DIR="$(cd $(dirname $0)/..; pwd)"
pushd $PROJ_DIR

cd python/pybackend_libs/
python3 setup.py bdist_wheel