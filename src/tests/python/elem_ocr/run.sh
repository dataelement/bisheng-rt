#!/bin/bash

curr=$(cd $(dirname $0); pwd)
pushd $curr
unset http_proxy
unset https_proxy


function index() {
  curl -X POST http://127.0.0.1:9001/v2/repository/index \
    -H 'Content-Type: application/json' -d '{}'
}

function model_status() {
  model="$1"
  curl -v -X GET http://127.0.0.1:9001/v2/models/${model}/ready \
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
# load_model elem_ocr_v3
# model_status elem_ocr_v3
# index

index
load_model elem_ocr_collection_v3
# model_status elem_ocr_collection_v3
# unload_model elem_ocr_collection_v3
index

# infer elem_ocr_v3 "model_req.json"
# unload_model "elem_ocr_v3"l
