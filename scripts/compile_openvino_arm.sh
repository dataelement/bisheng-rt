#!/bin/bash

git config --global http.proxy http://192.168.106.8:1081
git config --global https.proxy http://192.168.106.8:1081

function install_pkg() {
    apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-dev \
        libtbb-dev \
        patchelf \
        make \
        build-essential \
        wget \
        ca-certificates \
        libusb-1.0-0-dev
}

function build_2021() {
  OPENVINO_BUILD_TYPE="Release"
  # OPENVINO_VERSION="2021.4"

  OPENVINO_VERSION="2022.2.0"
  cd /workspace

  # git clone -b ${OPENVINO_VERSION} https://github.com/openvinotoolkit/openvino.git
  # cd /workspace/openvino
  # git submodule update --init --recursive
 
  # exit 0
  mkdir /workspace/openvino/build
  cd  /workspace/openvino/build
  # cmake \
  #    -DCMAKE_BUILD_TYPE=${OPENVINO_BUILD_TYPE} \
  #    -DCMAKE_INSTALL_PREFIX=/workspace/install \
  #    -DENABLE_VPU=OFF \
  #    -DENABLE_CLDNN=OFF \
  #    -DTHREADING=OMP \
  #    -DENABLE_GNA=OFF \
  #    -DENABLE_DLIA=OFF \
  #    -DENABLE_TESTS=OFF \
  #    -DENABLE_VALIDATION_SET=OFF \
  #    -DNGRAPH_ONNX_IMPORT_ENABLE=OFF \
  #    -DNGRAPH_DEPRECATED_ENABLE=FALSE \
  #    .. && \
  #    make -j$(nproc) install

    # TEMPCV_DIR=/workspace/openvino/inference-engine/temp/opencv_4* && \
    # OPENCV_DIRS=$(ls -d -1 ${TEMPCV_DIR}) && \
    # export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${OPENCV_DIRS[0]}/opencv/lib && \

  # exit 0
  mkdir /opt/openvino
  cd /opt/openvino
  IPREFIX=/workspace/install/deployment_tools/inference_engine/lib/intel64
  cp -r /workspace/openvino/licensing LICENSE.openvino
  cp -r /workspace/openvino/inference-engine/include .
  mkdir -p lib && \
    cp ${IPREFIX}/libinference_engine.so lib/. && \
    cp ${IPREFIX}/libinference_engine_legacy.so lib/. && \
    cp ${IPREFIX}/libinference_engine_transformations.so lib/. && \
    cp ${IPREFIX}/libinference_engine_lp_transformations.so lib/. && \
    cp ${IPREFIX}/libinference_engine_ir_reader.so lib/. && \
    cp ${IPREFIX}/libMKLDNNPlugin.so lib/. && \

  cp /workspace/install/deployment_tools/ngraph/lib/libngraph.so lib/. && \
    cp /workspace/install/deployment_tools/inference_engine/external/omp/lib/libiomp5.so lib/.

  cd lib && \
     for i in `find . -mindepth 1 -maxdepth 1 -type f -name '*\.so*'`; do \
        patchelf --set-rpath '$ORIGIN' $i; \
     done

}


function build_arm_2022() {
  OPENVINO_BUILD_TYPE="Release"
  # OPENVINO_VERSION="2021.4"

  OPENVINO_VERSION="2022.2.0"
  cd /workspace

  # git clone -b ${OPENVINO_VERSION} https://github.com/openvinotoolkit/openvino.git
  # cd /workspace/openvino
  # git submodule update --init --recursive
 
  # exit 0
  mkdir /workspace/openvino/build
  cd  /workspace/openvino/build
  # cmake \
  #    -DCMAKE_BUILD_TYPE=${OPENVINO_BUILD_TYPE} \
  #    -DCMAKE_INSTALL_PREFIX=/workspace/install \
  #    -DENABLE_VPU=OFF \
  #    -DENABLE_CLDNN=OFF \
  #    -DTHREADING=OMP \
  #    -DENABLE_GNA=OFF \
  #    -DENABLE_DLIA=OFF \
  #    -DENABLE_TESTS=OFF \
  #    -DENABLE_INTEL_MYRIAD=OFF \
  #    -DENABLE_VALIDATION_SET=OFF \
  #    -DNGRAPH_ONNX_IMPORT_ENABLE=OFF \
  #    -DNGRAPH_DEPRECATED_ENABLE=FALSE \
  #    .. && \
  #    make -j$(nproc) install

    # TEMPCV_DIR=/workspace/openvino/inference-engine/temp/opencv_4* && \
    # OPENCV_DIRS=$(ls -d -1 ${TEMPCV_DIR}) && \
    # export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${OPENCV_DIRS[0]}/opencv/lib && \
 
  mkdir /workspace/openvino_contrib
  git clone --recurse-submodules --single-branch --branch=releases/2022/2 https://github.com/openvinotoolkit/openvino_contrib.git 
  cd openvino_contrib/modules/arm_plugin
  mkdir build && cd build
  cmake -DInferenceEngineDeveloperPackage_DIR=/workspace/openvino/build -DCMAKE_BUILD_TYPE=Release .. && make

  # exit 0
  mkdir /opt/openvino
  cd /opt/openvino

  cp -r /workspace/openvino/licensing LICENSE.openvino
  mkdir -p include && \
    cp -r /workspace/install/runtime/include/ie/* include/. && \
    cp -r /workspace/install/runtime/include/ngraph include/. && \
    cp -r /workspace/install/runtime/include/openvino include/.
  mkdir -p lib && \
    cp /workspace/install/runtime/lib/aarch64/*.so lib/. && \
    # cp /workspace/install/runtime/3rdparty/omp/lib/libiomp5.so lib/.

  cd lib && \
     for i in `find . -mindepth 1 -maxdepth 1 -type f -name '*\.so*'`; do \
        patchelf --set-rpath '$ORIGIN' $i; \
     done

}

# install_pkg
build_arm_2022