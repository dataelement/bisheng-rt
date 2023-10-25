#!/bin/bash


function patch_vllm() {
  VLLM_PKG_PATH="/usr/local/lib/python3.8/dist-packages/vllm"
  cp patches/vllm/sequence.py ${VLLM_PKG_PATH}/
}


function patch_code_geex2() {
  MODEL_REPO="/home/public/llm/codegeex2-6b"
  CHATGLM2_MODEL_REPO="/home/public/llm/chatglm2-6b"
  cp $CHATGLM2_MODEL_REPO/totokenization_chatglm.py $MODEL_REPO
}


# patch_vllm
