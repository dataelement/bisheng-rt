#!/bin/bash


function run_test() {
  pushd python/pybackend_libs/src
  # PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=6 python3 tests/test_layout_mrcnn.py
  PYTHONPATH=. python3 tests/test_layout_mrcnn.py
  popd
}


function run_container() {
  LOCAL_MODEL_REPO="/home/public/llm"
  MOUNT="-v $LOCAL_MODEL_REPO:/opt/bisheng-rt/models/model_repository"
  IMAGE="dataelement/bisheng-rt:0.0.2"
  docker run --gpus=all -p 9011:9001 -p 9012:9002 -itd --workdir /opt/bisheng-rt \
      --shm-size=10G --name bisheng_rt_v002 ${MOUNT} $IMAGE ./bin/rtserver f
}

run_container
# run_test