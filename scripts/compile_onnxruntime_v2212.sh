#!/bin/bash

git config --global http.proxy http://192.168.106.8:1081
git config --global https.proxy http://192.168.106.8:1081


function start_dev_container() { 
  echo "start dev container"
  MOUNT="-v $HOME:$HOME"
  IMAGE="dataelement/bisheng-rt-base:0.0.1"
  # IMAGE="dataelement/onnxruntime:22.12"
  docker run --gpus=all --net=host -itd --shm-size=10G --name libonnxruntime_2212_dev ${MOUNT} $IMAGE bash
}



function create_build_image() { 
  echo "create build image"
  apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        libcurl4-openssl-dev \
        libssl-dev \
        patchelf \
        gnupg1


  cd /workspace
  ONNXRUNTIME_VERSION=1.16.2
  ONNXRUNTIME_REPO=https://github.com/microsoft/onnxruntime

  git clone -b rel-${ONNXRUNTIME_VERSION} --recursive ${ONNXRUNTIME_REPO} onnxruntime && \
    (cd onnxruntime && git submodule update --init --recursive)

  # docker commit -a "author@dataelem.com" -m "commit onnxruntime dev image" libonnxruntime_2212_dev dataelement/onnxruntime:22.12
}


function compile_onnxruntime() {

  # Notice: https://onnxruntime.ai/docs/reference/compatibility.html
  #  https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html

  # patch 
  # pip3 install flatbuffers -i https://mirrors.tencent.com/pypi/simple

  # update cmake to 3.26
  # wget http://192.168.106.8:8100/deps/cmake-3.26.3-linux-x86_64.tar.gz
  # tar zxf cmake-3.26.3-linux-x86_64.tar.gz --strip-components 1 -C /usr/local
  # rm cmake-3.26.3-linux-x86_64.tar.gz
  # exit 0

  _CUDNN_VERSION=$(echo $CUDNN_VERSION | cut -d. -f1-2)
  ## make cundnn patch
  # mkdir -p /usr/local/cudnn-${_CUDNN_VERSION}/cuda/include && \
  #   ln -s /usr/include/cudnn.h /usr/local/cudnn-${_CUDNN_VERSION}/cuda/include/cudnn.h && \
  #   mkdir -p /usr/local/cudnn-${_CUDNN_VERSION}/cuda/lib64 && \
  #   ln -s /etc/alternatives/libcudnn_so /usr/local/cudnn-${_CUDNN_VERSION}/cuda/lib64/libcudnn.so


  # Build
  # EG_FLAGS
  EG_FLAGS="--use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/local/cudnn-${_CUDNN_VERSION}/cuda"
  ONNXRUNTIME_BUILD_CONFIG="Release"
  # SM="52;60;61;70;75;80;86;89;90"

  cd /workspace/onnxruntime
  COMMON_BUILD_ARGS="--config ${ONNXRUNTIME_BUILD_CONFIG} --skip_submodule_sync --parallel --build_shared_lib --build_dir /workspace/build --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES='52;60;61;70;75;80;86;89;90' "
  ./build.sh ${COMMON_BUILD_ARGS} --update --build ${EG_FLAGS} --allow_running_as_root

}

function release() {
  ONNXRUNTIME_BUILD_CONFIG="Release"

  # license
  mkdir -p /opt/onnxruntime && \
    cp /workspace/onnxruntime/LICENSE /opt/onnxruntime && \
    cat /workspace/onnxruntime/cmake/external/onnx/VERSION_NUMBER > /opt/onnxruntime/ort_onnx_version.txt

  # ONNX Runtime headers, libraries and binaries
  mkdir -p /opt/onnxruntime/include && \
      cp /workspace/onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h \
         /opt/onnxruntime/include && \
      cp /workspace/onnxruntime/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h \
         /opt/onnxruntime/include && \
      cp /workspace/onnxruntime/include/onnxruntime/core/providers/cpu/cpu_provider_factory.h \
         /opt/onnxruntime/include

  mkdir -p /opt/onnxruntime/lib && \
    cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/libonnxruntime_providers_shared.so \
       /opt/onnxruntime/lib && \
    cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/libonnxruntime.so \
       /opt/onnxruntime/lib

  mkdir -p /opt/onnxruntime/bin && \
    cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/onnxruntime_perf_test \
       /opt/onnxruntime/bin && \
    cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/onnx_test_runner \
       /opt/onnxruntime/bin && \
    (cd /opt/onnxruntime/bin && chmod a+x *)

  # ENABLE GPU
  cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/libonnxruntime_providers_cuda.so \
       /opt/onnxruntime/lib

  # patchelf
  cd /opt/onnxruntime/lib && \
    for i in `find . -mindepth 1 -maxdepth 1 -type f -name '*\.so*'`; do \
        patchelf --set-rpath '$ORIGIN' $i; \
    done

  # For testing copy ONNX custom op library and model
  mkdir -p /opt/onnxruntime/test && \
    cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/libcustom_op_library.so \
       /opt/onnxruntime/test && \
    cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/testdata/custom_op_library/custom_op_test.onnx \
       /opt/onnxruntime/test 

  # docker cp libonnxruntime_2212_dev:/opt/onnxruntime libonnxruntime-v22.12

}


# start_dev_container
# create_build_image
# compile_onnxruntime
release