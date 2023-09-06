#!/bin/bash

git config --global http.proxy http://192.168.106.8:1081
git config --global https.proxy http://192.168.106.8:1081

function install_pkg() {
	apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        wget \
        zip \
        ca-certificates \
        build-essential \
        curl \
        libcurl4-openssl-dev \
        libssl-dev \
        patchelf \
        python3-dev \
        python3-pip \
        git \
        gnupg gnupg1
}

function build() {
  ONNXRUNTIME_VERSION=1.12.0
  ONNXRUNTIME_REPO=https://github.com/microsoft/onnxruntime
  ONNXRUNTIME_BUILD_CONFIG="Release"
  ONNXRUNTIME_OPENVINO_VERSION=2021.4.582
  cd /workspace

  # make cundnn patch
  _CUDNN_VERSION=$(echo $CUDNN_VERSION | cut -d. -f1-2)
  # mkdir -p /usr/local/cudnn-${_CUDNN_VERSION}/cuda/include && \
  #   ln -s /usr/include/cudnn.h /usr/local/cudnn-${_CUDNN_VERSION}/cuda/include/cudnn.h && \
  #   mkdir -p /usr/local/cudnn-${_CUDNN_VERSION}/cuda/lib64 && \
  #   ln -s /etc/alternatives/libcudnn_so /usr/local/cudnn-${_CUDNN_VERSION}/cuda/lib64/libcudnn.so

  # Install OpenVINO
  # ARG ONNXRUNTIME_OPENVINO_VERSION
  export INTEL_OPENVINO_DIR=/opt/intel/openvino_${ONNXRUNTIME_OPENVINO_VERSION}
  export LD_LIBRARY_PATH=$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/lib/intel64:$INTEL_OPENVINO_DIR/deployment_tools/ngraph/lib:$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/external/tbb/lib:/usr/local/openblas/lib:$LD_LIBRARY_PATH
  export PYTHONPATH=$INTEL_OPENVINO_DIR/tools:$PYTHONPATH
  export IE_PLUGINS_PATH=$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/lib/intel64
  export InferenceEngine_DIR=$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/share
  export ngraph_DIR=$INTEL_OPENVINO_DIR/deployment_tools/ngraph/cmake

  # From 2021.3 onwards, install_openvino_dependencies defaults to enabling interactive mode.
  # We use -y to force non-interactive mode.
  # wget https://apt.repos.intel.com/openvino/2021/GPG-PUB-KEY-INTEL-OPENVINO-2021 && \
  #     apt-key add GPG-PUB-KEY-INTEL-OPENVINO-2021 && rm GPG-PUB-KEY-INTEL-OPENVINO-2021 && \
  #     cd /etc/apt/sources.list.d && \
  #     echo "deb https://apt.repos.intel.com/openvino/2021 all main">intel-openvino-2021.list && \
  #     apt update && \
  #     apt install -y intel-openvino-dev-ubuntu20-${ONNXRUNTIME_OPENVINO_VERSION} && \
  #     cd ${INTEL_OPENVINO_DIR}/install_dependencies && ./install_openvino_dependencies.sh -y
  
  # cd /workspace
  # PROXYOPT="-e https_proxy=http://192.168.106.8:1081"
  # INTEL_COMPUTE_RUNTIME_URL=https://github.com/intel/compute-runtime/releases/download/19.41.14441
  # wget $PROXYOPT ${INTEL_COMPUTE_RUNTIME_URL}/intel-gmmlib_19.3.2_amd64.deb && \
  #     wget $PROXYOPT ${INTEL_COMPUTE_RUNTIME_URL}/intel-igc-core_1.0.2597_amd64.deb && \
  #     wget $PROXYOPT ${INTEL_COMPUTE_RUNTIME_URL}/intel-igc-opencl_1.0.2597_amd64.deb && \
  #     wget $PROXYOPT ${INTEL_COMPUTE_RUNTIME_URL}/intel-opencl_19.41.14441_amd64.deb && \
  #     wget $PROXYOPT ${INTEL_COMPUTE_RUNTIME_URL}/intel-ocloc_19.41.14441_amd64.deb && \
  #     dpkg -i *.deb && rm -rf *.deb

  # ONNX Runtime build
  # cd /workspace
  # git clone -b rel-${ONNXRUNTIME_VERSION} --recursive ${ONNXRUNTIME_REPO} onnxruntime && \
  #     (cd onnxruntime && git submodule update --init --recursive)
 
  # EG_FLAGS
  EG_FLAGS="--use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/local/cudnn-${_CUDNN_VERSION}/cuda --use_openvino CPU_FP32"

  cd /workspace/onnxruntime
  COMMON_BUILD_ARGS="--config ${ONNXRUNTIME_BUILD_CONFIG} --skip_submodule_sync --parallel --build_shared_lib --build_dir /workspace/build --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES='52;60;61;70;75;80;86' "
  
  # sed -i 's/VERS_%s//' tools/ci_build/gen_def.py &&  (sed -i 's/% VERSION_STRING//' tools/ci_build/gen_def.py) 
  # sed -i 's/set_target_properties(onnxruntime PROPERTIES VERSION ${ORT_VERSION})//' cmake/onnxruntime.cmake
  # ./build.sh ${COMMON_BUILD_ARGS} --update --build ${EG_FLAGS}

  # exit 0
  
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

  # OpenVino specific headers and libraries
  cp -r /opt/intel/openvino_${ONNXRUNTIME_OPENVINO_VERSION}/licensing \
       /opt/onnxruntime/LICENSE.openvino

  cp /workspace/onnxruntime/include/onnxruntime/core/providers/openvino/openvino_provider_factory.h \
       /opt/onnxruntime/include

  # libonnx_proto.so, libprotobuf.so.3.7.1.0 are needed when openvino execution provider is used
  if [ -f /opt/intel/openvino_${ONNXRUNTIME_OPENVINO_VERSION}/deployment_tools/ngraph/lib/libonnx_proto.so ]; then \
    cp /opt/intel/openvino_${ONNXRUNTIME_OPENVINO_VERSION}/deployment_tools/ngraph/lib/libonnx_proto.so \
        /opt/onnxruntime/lib; \
      cp /opt/intel/openvino_${ONNXRUNTIME_OPENVINO_VERSION}/deployment_tools/ngraph/lib/libprotobuf.so.3.7.1.0 \
        /opt/onnxruntime/lib; \
      (cd /opt/onnxruntime/lib && ln -sf libprotobuf.so.3.7.1.0 libprotobuf.so); \
  fi

  cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/libonnxruntime_providers_openvino.so \
       /opt/onnxruntime/lib && \
    cp /opt/intel/openvino_${ONNXRUNTIME_OPENVINO_VERSION}/deployment_tools/inference_engine/lib/intel64/libinference_engine.so \
       /opt/onnxruntime/lib && \
    cp /opt/intel/openvino_${ONNXRUNTIME_OPENVINO_VERSION}/deployment_tools/inference_engine/lib/intel64/libinference_engine_legacy.so \
       /opt/onnxruntime/lib && \
    cp /opt/intel/openvino_${ONNXRUNTIME_OPENVINO_VERSION}/deployment_tools/inference_engine/lib/intel64/libinference_engine_transformations.so \
       /opt/onnxruntime/lib && \
    cp /opt/intel/openvino_${ONNXRUNTIME_OPENVINO_VERSION}/deployment_tools/ngraph/lib/libngraph.so \
       /opt/onnxruntime/lib && \
    cp /opt/intel/openvino_${ONNXRUNTIME_OPENVINO_VERSION}/deployment_tools/ngraph/lib/libonnx_importer.so \
       /opt/onnxruntime/lib && \
    cp /opt/intel/openvino_${ONNXRUNTIME_OPENVINO_VERSION}/deployment_tools/inference_engine/external/tbb/lib/libtbb.so.2 \
       /opt/onnxruntime/lib && \
    cp /opt/intel/openvino_${ONNXRUNTIME_OPENVINO_VERSION}/deployment_tools/inference_engine/lib/intel64/plugins.xml \
       /opt/onnxruntime/lib && \
    cp /opt/intel/openvino_${ONNXRUNTIME_OPENVINO_VERSION}/deployment_tools/inference_engine/lib/intel64/libMKLDNNPlugin.so \
       /opt/onnxruntime/lib && \
    cp /opt/intel/openvino_${ONNXRUNTIME_OPENVINO_VERSION}/deployment_tools/inference_engine/lib/intel64/libinference_engine_lp_transformations.so \
       /opt/onnxruntime/lib && \
    cp /opt/intel/openvino_${ONNXRUNTIME_OPENVINO_VERSION}/deployment_tools/inference_engine/lib/intel64/libinference_engine_ir_reader.so \
       /opt/onnxruntime/lib && \
    cp /opt/intel/openvino_${ONNXRUNTIME_OPENVINO_VERSION}/deployment_tools/inference_engine/lib/intel64/libinference_engine_onnx_reader.so \
       /opt/onnxruntime/lib && \
    (cd /opt/onnxruntime/lib && \
     chmod a-x * && \
     ln -sf libtbb.so.2 libtbb.so)

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
}

# install_pkg
build