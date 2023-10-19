#!/bin/bash

curr=$(cd $(dirname $0); pwd)
pushd $curr
unset http_proxy
unset https_proxy

function load_model() {
  model="$1"
  curl -v -X POST http://127.0.0.1:9011/v2/repository/models/${model}/load \
    -H 'Content-Type: application/json' -d @${model}.json  
}


function unload_model() {
  model="$1"
  curl -v -X POST http://127.0.0.1:9011/v2/repository/models/${model}/unload \
    -H 'Content-Type: application/json' -d '{}'
}


load_model "Qwen-14B-Chat"
# load_model "Qwen-7B-Chat"
# unload_model "Qwen-7B-Chat"
# python3 vllm_model_client.py -u 192.168.106.12:9010 -m "Qwen-7B-Chat"