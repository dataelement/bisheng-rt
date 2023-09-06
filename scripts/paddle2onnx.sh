#!/bin/bash

models=(  
  ArabicPPOCRV3_Rec
  ChPPOCRV2_Det
  ChPPOCRV2_Rec
  ChPPOCRV3_Det
  ChPPOCRV3_Rec
  ChPPOcrMobileV2_Cls
  ChPPOcrMobileV2_Det
  ChPPOcrMobileV2_Rec
  ChPPOcrServerV2_Det
  ChPPOcrServerV2_Rec
  ChtPPOCRV3_Rec
  CyrillicPPOCRV3_Rec
  DevanagariPPOCRV3_Rec
  EnNumberMobileV2_Rec
  EnPPOCRV3_Det
  EnPPOCRV3_Rec
  JapanPPOCRV3_Rec
  KaPPOCRV3_Rec
  KoreaPPOCRV3_Rec
  LatinPPOCRV3_Rec
  MlPPOCRV3_Det
  TaPPOCRV3_Rec
  TePPOCRV3_Rec
)

# models=(ArabicPPOCRV3_Rec)

function trans() {
    model="$1"
    output_dir=$2
    paddle2onnx --model_dir ./graphs/${model}/1 \
      --model_filename model.pdmodel \
      --params_filename model.pdiparams \
      --save_file ${output_dir}/model.onnx \
      --opset_version 11 \
      --input_shape_dict="{'x':[-1,3,-1,-1]}" \
      --enable_onnx_checker True
}

function run() {
  output_prefix="./graphs_onnx"
  for m in ${models[@]}; do
    echo $m
    output_dir=${output_prefix}/${m}/1
    if [[ ! -d ${output_dir} ]]; then
      mkdir -p ${output_dir}
    fi

    trans $m $output_dir
  done
}

run