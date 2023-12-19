#!/bin/bash


pushd $(cd $(dirname $0); pwd)
export RT_EP="192.168.106.127:9001"
export PYTHONPATH="/home/hanfeng/projects/bisheng-rt/python/pybackend_libs/src"
python3 test_layout_with_ocr.py


