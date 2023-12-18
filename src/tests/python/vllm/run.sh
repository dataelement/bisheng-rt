#!/bin/bash

curr=$(cd $(dirname $0); pwd)
pushd $curr
unset http_proxy
unset https_proxy


function index() {
  curl -X POST http://127.0.0.1:9001/v2/repository/index \
    -H 'Content-Type: application/json' -d '{}'
}


function load_model() {
  model="$1"
  curl -v -X POST http://127.0.0.1:9001/v2/repository/models/${model}/load \
    -H 'Content-Type: application/json' -d @${model}.json  
}


function unload_model() {
  model="$1"
  curl -v -X POST http://127.0.0.1:9001/v2/repository/models/${model}/unload \
    -H 'Content-Type: application/json' -d '{}'
}


function load_nondecoupled_model() {
  model="$1"
  curl -v -X POST http://127.0.0.1:9001/v2/repository/models/${model}/load \
    -H 'Content-Type: application/json' -d @${model}-nondecoupled.json 
}


function infer() {
  model="$1"
  req="$2"
  curl -v -X POST http://127.0.0.1:9001/v2.1/models/${model}/infer \
    -H 'Content-Type: application/json' -d @${req}
}


# index
# load_model "Qwen-1_8B-Chat"
# unload_model "Qwen-1_8B-Chat"
# infer "Qwen-1_8B-Chat" "model_req.json"
# index

# load_model "Qwen-14B-Chat"
# load_model "Qwen-7B-Chat"

# load_nondecoupled_model "Qwen-7B-Chat"
# infer "Qwen-7B-Chat" "model_req.json"

# load_model "Qwen-7B-Chat"
# python3 vllm_model_client.py -u 192.168.106.12:19000 -m Qwen-7B-Chat -s
# unload_model "Qwen-7B-Chat"