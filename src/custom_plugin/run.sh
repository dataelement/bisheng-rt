#!/bin/bash

# git config --global http.proxy http://192.168.106.8:1081
# git config --global https.proxy http://192.168.106.8:1081

PROJ_DIR="/home/hanfeng/projects/idpserver"
THIRD_PARTY_DIR="${PROJ_DIR}/tritonbuild/tritonserver/build/third-party"
ABSL_DIR="${THIRD_PARTY_DIR}/absl/lib/cmake/absl"

function build_daliop() {
  name="${1}"
  src_dir="${PROJ_DIR}/src/custom_plugin/${name}/"
  build_dir="${PROJ_DIR}/tritonbuild/custom_plugin/${name}/build"
  install_dir="${PROJ_DIR}/tritonbuild/custom_plugin/${name}/install"
  output_dir="${PROJ_DIR}/tritonbuild/install/plugins/dali"
  if [ ! -d ${build_dir} ]; then
    mkdir -p ${build_dir}
  fi

  if [ ! -d ${output_dir} ]; then
    mkdir -p ${output_dir}
  fi

  pushd ${build_dir}
  cmake ${src_dir}                              \
    -DCMAKE_PREFIX_PATH=${ABSL_DIR}             \
    -DCMAKE_INSTALL_PREFIX:PATH=${install_dir}  \
    -DPROJ_ROOT_DIR:PATH=${PROJ_DIR} && \
    make -j$(nproc)
  cp ${build_dir}/libcustomops.so ${output_dir}/
  popd
}

function build_tflib() {
  name="tf"
  src_dir="${PROJ_DIR}/src/custom_plugin/${name}/"
  build_dir="${PROJ_DIR}/tritonbuild/custom_plugin/${name}/build"
  install_dir="${PROJ_DIR}/tritonbuild/custom_plugin/${name}/install"
  output_dir="${PROJ_DIR}/tritonbuild/install/plugins/tf"
  TF_PATH="${PROJ_DIR}/tritonbuild/third_party/libtf2_v22.08"
  PROTOBUF_INC_DIR="${TF_PATH}/include"
  if [ ! -d ${build_dir} ]; then
    mkdir -p ${build_dir}
  fi

  if [ ! -d ${output_dir} ]; then
    mkdir -p ${output_dir}
  fi

  # patch for the cuda path in third party
  if [ ! -d ${TF_PATH}/include/third_party/gpus/cuda ]; then
    mkdir -p ${TF_PATH}/include/third_party/gpus/cuda
    ln -s /usr/local/cuda-11.7/targets/x86_64-linux/include \
      ${TF_PATH}/include/third_party/gpus/cuda/include
  fi

  pushd ${build_dir}
  cmake ${src_dir}                             \
    -DTF_VER=2                                 \
    -DCMAKE_BUILD_TYPE=Release                 \
    -DBUILD_TF_CC=OFF                          \
    -DBUILD_TF=ON                              \
    -DCUDA_VERSION=11.7                        \
    -DTF_PATH=${TF_PATH}                       \
    -DPROTOBUF_INC_DIR=${PROTOBUF_INC_DIR} &&  \
    make -j$(nproc)
  cp ${build_dir}/lib/libtf_fastertransformer_op.so ${output_dir}/
  popd
}

function build_trtlib() {
  name="trt"
  src_dir="${PROJ_DIR}/src/custom_plugin/${name}/"
  build_dir="${PROJ_DIR}/tritonbuild/custom_plugin/${name}/build"
  install_dir="${PROJ_DIR}/tritonbuild/custom_plugin/${name}/install"
  output_dir="${PROJ_DIR}/tritonbuild/install/plugins"
  if [ ! -d ${build_dir} ]; then
    mkdir -p ${build_dir}
  fi

  pushd ${build_dir}
  cmake ${src_dir}                                 \
    -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu        \
    -DTRT_INC_DIR=/usr/include/x86_64-linux-gnu    \
    -DTRT_BIN_DIR=${output_dir}/trt                \
    -DCUDA_VERSION=11.7 -DCUDNN_VERSION=8.5        \
    -DTHIRD_PARTY_PREFIX=${build_dir}/third_party  \
    -DCMAKE_INSTALL_PREFIX:PATH=${install_dir} &&  \
    make -j$(nproc) && make install
  popd
}


# build_daliop
# build_trtlib
build_tflib