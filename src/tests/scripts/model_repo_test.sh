#!/bin/bash


function up_repo() {
  cp -fr resource/internal_model_repository/* \
    ./tritonbuild/install/resource/internal_model_repository/
}


function index_model() {
  curl -X POST http://192.168.106.12:8502/v2/repository/index \
   -H 'Content-Type: application/json' \
   -d '{}'
}


function generate_load_data() {
  cat <<EOF
{
  "parameters": {
    "type": "dataelem.pymodel.huggingface_model",
    "instance_groups": "device=gpu;gpus=0,1|2"
  }
}
EOF
}

function load_model() {
  curl -v -X POST http://192.168.106.12:8502/v2/repository/models/test/load \
   -H 'Content-Type: application/json' \
   -d "$(generate_load_data)"
}


function unload_model() {
  curl -X POST http://192.168.106.12:8502/v2/repository/models/test/unload \
   -H 'Content-Type: application/json' \
   -d '{}'
}


function model_ready() {
  curl -v -X GET http://192.168.106.12:8502/v2/models/test2/ready \
   -H 'Content-Type: application/json' \
   -d '{}'
}


function model_infer() {
  curl -v -X POST http://192.168.106.12:8502/v2.1/models/test/infer \
   -H 'Content-Type: application/json' \
   -d '{"texts":[1]}'
}


up_repo
# index_model

load_model
index_model

# unload_model
# index_model

# model_ready

# model_infer