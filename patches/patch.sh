#!/bin/bash


function patch_vllm() {
  VLLM_PKG_PATH="/usr/local/lib/python3.8/dist-packages/vllm"
  cp patches/vllm/sequence.py ${VLLM_PKG_PATH}/
}


patch_vllm
