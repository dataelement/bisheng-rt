#!/bin/bash


function up_repo() {
  cp -fr resource/internal_model_repository/* \
    ./tritonbuild/install/resource/internal_model_repository/

  cp -fr python/pybackend_libs/src/pybackend_libs ./tritonbuild/install/backends/python/
}


function index_model() {
  curl -X POST http://192.168.106.12:7001/v2/repository/index \
   -H 'Content-Type: application/json' \
   -d '{}'
}


function load_data1() {
  cat <<EOF
{
  "parameters": {
    "type": "dataelem.pymodel.huggingface_model",
    "pymodel_type": "embedding.ME5Embedding",
    "gpu_memory": "5",
    "instance_groups": "device=gpu;gpus=7"
  }
}
EOF
}

function load_data2() {
  cat <<EOF
{
  "parameters": {
    "type": "dataelem.pymodel.huggingface_model",
    "pymodel_type": "llm.Llama2Chat",
    "gpu_memory": "36",
    "instance_groups": "device=gpu;gpus=7,8",
    "max_tokens": "4096"
  }
}
EOF
}

function load_model2() {
  model="$1"
  curl -v -X POST http://192.168.106.12:7001/v2/repository/models/${model}/load \
   -H 'Content-Type: application/json' \
   -d "$(load_data2)"
}

function load_model1() {
  model="$1"
  curl -v -X POST http://192.168.106.12:7001/v2/repository/models/${model}/load \
   -H 'Content-Type: application/json' \
   -d "$(load_data1)"
}


function unload_model() {
  model="$1"
  curl -v -X POST http://192.168.106.12:7001/v2/repository/models/${model}/unload \
   -H 'Content-Type: application/json' \
   -d '{}'
}


function model_ready() {
  curl -v -X GET http://192.168.106.12:7001/v2/models/test2/ready \
   -H 'Content-Type: application/json' \
   -d '{}'
}


function infer_data() {
  cat <<EOF
{

  "model": "multilingual-e5-large",
  "texts": ["how much protein should a female eat"],
  "type": "query"
}
EOF
}


function infer_data2() {
  cat <<EOF
{

  "model": "Llama-2-13b-chat-hf",
  "messages": [
    {"role": "user", "content": "hello"}
   ]
}
EOF
}


function model_infer() {
  model="$1"
  curl -v -X POST http://192.168.106.12:7001/v2.1/models/${model}/infer \
   -H 'Content-Type: application/json' \
   -d "$(infer_data2)"
}

m1="multilingual-e5-large"
m2="Llama-2-13b-chat-hf"



case $1 in
  update)
    echo -n "update"
    up_repo
    index_model
    ;;
  load)
    echo -n "load"
    load_model2 "$m2"
    # load_model1 "$m1"
    index_model
    ;;
  unload)
    echo -n "unload"
    unload_model "$m2"
    ;;
  infer)
    echo -n "infer"
    model_infer "$m2"
    ;;
  *)
    echo -n "unknown"
    ;;
esac
