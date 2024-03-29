#!/bin/bash


function run_test() {
  pushd python/pybackend_libs/src
  # PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=6 python3 tests/test_layout_mrcnn.py
  PYTHONPATH=. python3 tests/test_layout_mrcnn.py
  popd
}


function run_container_dev() {
  LOCAL_MODEL_REPO="/home/public/llm"
  MOUNT="-v $LOCAL_MODEL_REPO:$LOCAL_MODEL_REPO -v $HOME:$HOME"
  IMAGE="dataelement/bisheng-rt:0.0.4.alpha1"
  docker run --gpus=all --net=host -itd --workdir /opt/bisheng-rt \
      --shm-size=10G --name bisheng_rt_v004_dev ${MOUNT} $IMAGE bash
}


function run_container() {
  LOCAL_MODEL_REPO="/home/public/llm"
  MAPING_MODEL_REPO="/opt/bisheng-rt/models/model_repository"
  MOUNT="-v $LOCAL_MODEL_REPO:$MAPING_MODEL_REPO -v $HOME:$HOME"
  IMAGE="dataelement/bisheng-rt:0.0.2"
  docker run --gpus=all -p 9001:9001 -p 9002:9002 -itd --workdir /opt/bisheng-rt \
      --shm-size=10G --name bisheng_rt_v002 ${MOUNT} $IMAGE ./bin/rtserver f
}


function run_container_v003() {
  LOCAL_MODEL_REPO="/home/public/llm"
  MAPING_MODEL_REPO="/opt/bisheng-rt/models/model_repository"
  MOUNT="-v $LOCAL_MODEL_REPO:$MAPING_MODEL_REPO -v $HOME:$HOME"
  IMAGE="dataelement/bisheng-rt:0.0.3"
  docker run --gpus=all -p 9000:9000 -p 9001:9001 -p 9002:9002 -itd --workdir /opt/bisheng-rt \
      --shm-size=10G --name bisheng_rt_v003 ${MOUNT} $IMAGE bash
}

function run_container_v006() {

  LOCAL_RT_DEV="/public/bisheng/dev_workspace/bisheng-rt-006"
  LOCAL_MODEL_REPO="/public/bisheng/model_repository/"
  MAPING_MODEL_REPO="/opt/bisheng-rt/models/model_repository"
  MOUNT="-v ${LOCAL_RT_DEV}:/opt/bisheng-rt -v $LOCAL_MODEL_REPO:$MAPING_MODEL_REPO -v $HOME:$HOME -v /public:/public"
  IMAGE="dataelement/bisheng-rt-runtime:0.0.2"
  docker run --gpus=all --net=host -itd --workdir /opt/bisheng-rt \
      --shm-size=10G --name bisheng_rt_test_v006 ${MOUNT} $IMAGE bash
}


function run_container_v006_test() {
  LOCAL_MODEL_REPO="/public/bisheng/model_repository/"
  MAPING_MODEL_REPO="/opt/bisheng-rt/models/model_repository"
  MOUNT="-v $LOCAL_MODEL_REPO:$MAPING_MODEL_REPO -v $HOME:$HOME -v /public:/public"
  IMAGE="cr.dataelem.com/dataelement/bisheng-rt:0.0.6"
  docker run --gpus=all --net=host -itd --shm-size=10G \
    --name bisheng_rt_test_v006 ${MOUNT} $IMAGE bash
}


function run_container_v006_cpu_test() {
  LOCAL_MODEL_REPO="/public/bisheng/model_repository/"
  MAPING_MODEL_REPO="/opt/bisheng-rt/models/model_repository"
  MOUNT="-v $LOCAL_MODEL_REPO:$MAPING_MODEL_REPO -v $HOME:$HOME -v /public:/public"
  IMAGE="cr.dataelem.com/dataelement/bisheng-rt:0.0.6"
  docker run --net=host -itd --shm-size=10G \
    --name bisheng_rt_cpu_test_v006 ${MOUNT} $IMAGE bash
}

function run_container_v0065_test() {
  LOCAL_MODEL_REPO="/public/bisheng/model_repository/"
  MAPING_MODEL_REPO="/opt/bisheng-rt/models/model_repository"
  MOUNT="-v $LOCAL_MODEL_REPO:$MAPING_MODEL_REPO -v $HOME:$HOME -v /public:/public"
  IMAGE="cr.dataelem.com/dataelement/bisheng-rt:0.0.6.3"
  docker run --gpus=all --net=host -itd --shm-size=10G \
    --name bisheng_rt_test_v0065 ${MOUNT} $IMAGE bash
}

function post_install_r0064() {
  # Install vLLM with CUDA 11.8.
  VLLM_VERSION=0.3.3
  PYTHON_VERSION=38
  REPO="https://mirrors.aliyun.com/pypi/simple"
  # pip3 uninstall -y bisheng-pybackend-libs
  # pip3 install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl
  # pip3 install /public/bisheng/release/vllm-0.3.3+cu118-cp38-cp38-manylinux1_x86_64.whl -i ${REPO}
  pip3 uninstall -y cupy-cuda12x
  pip3 install cupy-cuda11x==12.1.0 -i ${REPO}
  # pip3 install lib/bisheng_pybackend_libs-0.0.3-py3-none-any.whl -i ${REPO}
}


function run_container_v0065() {
  LOCAL_MODEL_REPO="/public/bisheng/model_repository/"
  MAPING_MODEL_REPO="/opt/bisheng-rt/models/model_repository"
  MOUNT="-v $LOCAL_MODEL_REPO:$MAPING_MODEL_REPO -v $HOME:$HOME -v /public:/public"
  IMAGE="cr.dataelem.com/dataelement/bisheng-rt:0.0.6.5"
  docker run --gpus=all --net=host -itd --shm-size=10G \
    --name bisheng_rt_v0065 ${MOUNT} $IMAGE bash
}

run_container_v0065
# post_install_r0064
# run_container_v0065_test
# run_container_v006_cpu_test
# run_container_v006_test
# run_container_v006
# run_container_dev
# run_container
# run_container_v003
# run_test
