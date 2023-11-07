#!/bin/bash


pushd $(cd $(dirname $0); pwd)
export RT_EP="192.168.106.12:9005"
export PYTHONPATH="/home/hanfeng/projects/bisheng-rt/python/pybackend_libs/src"
python3 test_layout_with_ocr.py


