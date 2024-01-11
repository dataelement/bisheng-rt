#!/bin/bash

function test_recog() {
    PYLIB_PATH=${WORKDIR}/projects/bisheng-rt/python/pybackend_libs/src
    export PYTHONPATH=${PYLIB_PATH}
    python3 test_latex_recog.py
}


test_recog
# RT_EP=192.168.106.20:19001 python3 test_e2e.py