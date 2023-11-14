#!/bin/bash


function get_compute_cap() {
  out=$(nvidia-smi --query-gpu=compute_cap --format=csv)
  compute_cap=$(echo $out |awk '{print $2}')
  compute_cap=${compute_cap/\./}
  echo ${compute_cap}
}


function update_plan_model() {
  curr=$(cd $(dirname $0); pwd)
  models=(
    "transformer-hand-v1.16_trt"
    "transformer-blank-v0.2_trt"
  )

  local comp_cap=$(get_compute_cap)
  link_model_file="model.plan.pri"
  src_model_file="model.plan.pri.cc${comp_cap}"
  for model in ${models[@]}; do
    echo $model
    model_path=${curr}/$model/1
    pushd $model_path
    echo $src_model_file
    if [ -f  $link_model_file ]; then
      rm $link_model_file
    fi
    ln -s $src_model_file $link_model_file
    popd
  done
}


update_plan_model