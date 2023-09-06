#!/bin/bash

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
    "gpu_memory": "30",
    "instance_groups": "device=gpu;gpus=7,8"
  }
}
EOF
}

function load_model2() {
  model="Llama-2-13b-chat-hf"
  curl -v -X POST http://192.168.106.12:7001/v2/repository/models/${model}/load \
   -H 'Content-Type: application/json' \
   -d "$(load_data2)"
}

function load_model1() {
  model="multilingual-e5-large"
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

unload_model "multilingual-e5-large"

# load_model1
# load_model2