#!/bin/bash


function up_repo() {
  cp -fr resource/internal_model_repository/* \
    ./tritonbuild/install/resource/internal_model_repository/

  cp -fr python/pybackend_libs/src/pybackend_libs ./tritonbuild/install/backends/python/
}


function index_model() {
  curl -X POST http://192.168.106.12:8502/v2/repository/index \
   -H 'Content-Type: application/json' \
   -d '{}'
}


function load_data() {
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

function load_model() {
  model="$1"
  curl -v -X POST http://192.168.106.12:8502/v2/repository/models/${model}/load \
   -H 'Content-Type: application/json' \
   -d "$(load_data)"
}


function unload_model() {
  model="$1"
  curl -v -X POST http://192.168.106.12:8502/v2/repository/models/${model}/unload \
   -H 'Content-Type: application/json' \
   -d '{}'
}


function model_ready() {
  curl -v -X GET http://192.168.106.12:8502/v2/models/test2/ready \
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


function model_infer() {
  model="$1"
  curl -v -X POST http://192.168.106.12:8502/v2.1/models/${model}/infer \
   -H 'Content-Type: application/json' \
   -d "$(infer_data)"
}



case $1 in
  update)
    echo -n "update"
    up_repo
    index_model
    ;;
  load)
    echo -n "load"
    load_model "multilingual-e5-large"
    index_model
    ;;
  unload)
    echo -n "unload"
    unload_model "multilingual-e5-large"
    index_model
    ;;
  infer)
    echo -n "infer"
    model_infer "multilingual-e5-large"
    ;;
  *)
    echo -n "unknown"
    ;;
esac
