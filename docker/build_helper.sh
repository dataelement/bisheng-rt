#!/bin/bash

git config --global http.proxy http://192.168.106.8:1081
git config --global https.proxy http://192.168.106.8:1081

# export https_proxy=http://192.168.106.8:1081
# export http_proxy=http://192.168.106.8:1081

PROJ_DIR="$(cd $(dirname $0)/..; pwd)"
pushd $PROJ_DIR

THIRD_PARTY_DIR="${PROJ_DIR}/tritonbuild/tritonserver/build/third-party"
ABSL_DIR="${THIRD_PARTY_DIR}/absl/lib/cmake/absl"


function build_tf_backend() {
  BACKEND="$1"
  ver=${BACKEND//tensorflow/libtf}
  LIBTF_DIR=${PROJ_DIR}/tritonbuild/third_party/${ver}_v22.12

  python3 ./build.py \
    --backend=${BACKEND} \
    --no-container-build --no-core-build \
    --enable-gpu --enable-stats \
    --version 2.27.0 --container-version 22.12 --upstream-container-version 22.12 \
    --extra-backend-cmake-arg=${BACKEND}:TRITON_TENSORFLOW_LIB_PATHS=${LIBTF_DIR}/lib \
    --extra-backend-cmake-arg=${BACKEND}:TRITON_TENSORFLOW_INCLUDE_PATHS=${LIBTF_DIR}/include \
    --extra-backend-cmake-arg=${BACKEND}:TRITON_SRC_DIR=${PROJ_DIR}/src \
    --extra-backend-cmake-arg=${BACKEND}:absl_DIR:PATH=${ABSL_DIR} \
    --extra-backend-cmake-arg=${BACKEND}:TRITON_TENSORFLOW_INSTALL_EXTRA_DEPS=ON  \
    --cmake-dir ${PROJ_DIR} \
    --build-dir ${PROJ_DIR}/tritonbuild/backends \
    --install-dir ${PROJ_DIR}/tritonbuild/install
}

function build_pt_backend() {
  LIBPT_DIR=${PROJ_DIR}/tritonbuild/third_party/libtorch_v22.12
  BACKEND=pytorch
  python3 ./build.py -v \
    --backend=${BACKEND} \
    --no-container-build --no-core-build \
    --enable-gpu --enable-stats \
    --version 2.27.0 --container-version 22.12 --upstream-container-version 22.12 \
    --extra-backend-cmake-arg=${BACKEND}:TRITON_SRC_DIR=${PROJ_DIR}/src \
    --extra-backend-cmake-arg=${BACKEND}:TRITON_PYTORCH_LIB_PATHS=${LIBPT_DIR}/lib \
    --extra-backend-cmake-arg="${BACKEND}:TRITON_PYTORCH_INCLUDE_PATHS=${LIBPT_DIR}/include;${LIBPT_DIR}/include/torch" \
    --extra-backend-cmake-arg=${BACKEND}:absl_DIR:PATH=${ABSL_DIR} \
    --cmake-dir ${PROJ_DIR} \
    --build-dir ${PROJ_DIR}/tritonbuild/backends \
    --install-dir ${PROJ_DIR}/tritonbuild/install
}

function build_python_backend() {
  BACKEND=python
  python3 ./build.py -v \
    --backend=${BACKEND} \
    --no-container-build --no-core-build \
    --enable-gpu --enable-stats \
    --version 2.27.0 --container-version 22.12 --upstream-container-version 22.12 \
    --extra-backend-cmake-arg=${BACKEND}:TRITON_SRC_DIR=${PROJ_DIR}/src \
    --extra-backend-cmake-arg=${BACKEND}:absl_DIR:PATH=${ABSL_DIR} \
    --cmake-dir ${PROJ_DIR} \
    --build-dir ${PROJ_DIR}/tritonbuild/backends \
    --install-dir ${PROJ_DIR}/tritonbuild/install
}

function build_onnx_backend() {
  LIBPT_DIR=${PROJ_DIR}/tritonbuild/third_party/libonnxruntime-v22.12
  BACKEND=onnxruntime
  python3 ./build.py -v \
    --backend=${BACKEND} \
    --no-container-build --no-core-build \
    --enable-gpu --enable-stats \
    --version 2.27.0 --container-version 22.12 --upstream-container-version 22.12 \
    --extra-backend-cmake-arg=${BACKEND}:TRITON_ENABLE_ONNXRUNTIME_OPENVINO=OFF \
    --extra-backend-cmake-arg=${BACKEND}:TRITON_SRC_DIR=${PROJ_DIR}/src \
    --extra-backend-cmake-arg=${BACKEND}:absl_DIR:PATH=${ABSL_DIR} \
    --extra-backend-cmake-arg=${BACKEND}:TRITON_ONNXRUNTIME_LIB_PATHS=${LIBPT_DIR}/lib \
    --extra-backend-cmake-arg="${BACKEND}:TRITON_ONNXRUNTIME_INCLUDE_PATHS=${LIBPT_DIR}/include" \
    --cmake-dir ${PROJ_DIR} \
    --build-dir ${PROJ_DIR}/tritonbuild/backends \
    --install-dir ${PROJ_DIR}/tritonbuild/install

  cp -P ${LIBPT_DIR}/lib/* ${PROJ_DIR}/tritonbuild/install/backends/${BACKEND}/
}

function build_paddle_backend() {
  LIBPT_DIR=${PROJ_DIR}/tritonbuild/third_party/libpaddle_v22.08/paddle
  BACKEND=paddle
  python3 ./build.py -v \
    --backend=${BACKEND} \
    --no-container-build --no-core-build \
    --version 2.27.0 --container-version 22.12 --upstream-container-version 22.12 \
    --enable-gpu --enable-stats \
    --extra-backend-cmake-arg=${BACKEND}:TRITON_SRC_DIR=${PROJ_DIR}/src \
    --extra-backend-cmake-arg=${BACKEND}:absl_DIR:PATH=${ABSL_DIR} \
    --extra-backend-cmake-arg=${BACKEND}:PADDLE_LIB_PATHS=${LIBPT_DIR}/lib \
    --extra-backend-cmake-arg="${BACKEND}:PADDLE_INCLUDE_PATHS=${LIBPT_DIR}/include" \
    --cmake-dir ${PROJ_DIR} \
    --build-dir ${PROJ_DIR}/tritonbuild/backends \
    --install-dir ${PROJ_DIR}/tritonbuild/install
}

function build_tensorrt_backend() {
  BACKEND=tensorrt
  python3 ./build.py -v \
    --backend=${BACKEND} \
    --no-container-build --no-core-build \
    --version 2.27.0 --container-version 22.12 --upstream-container-version 22.12 \
    --enable-gpu --enable-stats --enable-nvtx \
    --extra-backend-cmake-arg=${BACKEND}:TRITON_SRC_DIR=${PROJ_DIR}/src \
    --extra-backend-cmake-arg=${BACKEND}:absl_DIR:PATH=${ABSL_DIR} \
    --cmake-dir ${PROJ_DIR} \
    --build-dir ${PROJ_DIR}/tritonbuild/backends \
    --install-dir ${PROJ_DIR}/tritonbuild/install
}


function build_dali_backend() {
  export PATH=/opt/conda/bin/:$PATH
  # conda env create \
  #   -f tritonbuild/backends/dali/build/src/dali_executor/dalienv.yml \
  #   -p tritonbuild/install/backends/dali/conda/envs/dalienv
  # exit 0

  BACKEND=dali
  python3 ./build.py -v \
    --backend=${BACKEND} \
    --no-container-build --no-core-build \
    --version 2.27.0 --container-version 22.08 --upstream-container-version 22.08 \
    --enable-gpu --enable-stats --enable-nvtx --enable-metrics \
    --extra-backend-cmake-arg=${BACKEND}:TRITON_SRC_DIR=${PROJ_DIR}/src \
    --extra-backend-cmake-arg=${BACKEND}:absl_DIR:PATH=${ABSL_DIR} \
    --cmake-dir ${PROJ_DIR} \
    --build-dir ${PROJ_DIR}/tritonbuild/backends \
    --install-dir ${PROJ_DIR}/tritonbuild/install

  # name=dali_backend
  # _dir="${PROJ_DIR}/src/backends/$name"
  # TRITON_BACKEND_API_VERSION="r22.08"
  # pushd ${_dir}
  # rm -fr build && mkdir build && cd build
  # cmake ..                                                             \
  #   -DCMAKE_PREFIX_PATH=${ABSL_DIR}                                    \
  #   -DCMAKE_INSTALL_PREFIX=`pwd`/install                               \
  #   -DCMAKE_BUILD_TYPE=Release                                         \
  #   -DTRITON_BACKEND_API_VERSION=${TRITON_BACKEND_API_VERSION}         \
  #   ${DALI_DOWNLOAD_PKG_NAME:+                                         \
  #     -DDALI_DOWNLOAD_PKG_NAME=${DALI_DOWNLOAD_PKG_NAME}}              \
  #   ${DALI_DOWNLOAD_EXTRA_INDEX_URL:+                                  \
  #     -DDALI_EXTRA_INDEX_URL=${DALI_DOWNLOAD_EXTRA_INDEX_URL}}         \
  #   -DDALI_VERSION=${DALI_DOWNLOAD_VERSION}                            \
  #   -DDALI_DOWNLOAD_EXTRA_OPTIONS="${DALI_DOWNLOAD_EXTRA_OPTIONS}" &&  \
  #   make -j $(nproc) && make install
  # cd ..
  # cp -fr build/install/backends/$name ${PROJ_DIR}/tritonbuild/install/backends/
}

function build_openvino_arm_backend() {
  # LIBPT_DIR=${PROJ_DIR}/tritonbuild/third_party/libopenvino_v22.08_2021.4
  LIBPT_DIR=${PROJ_DIR}/tritonbuild/third_party/libopenvino_v22.08_2022.2
  BACKEND=openvino
  python3 ./build.py -v \
    --backend=${BACKEND} \
    --no-container-build --no-core-build \
    --enable-gpu --enable-stats \
    --version 2.27.0 --container-version 22.08 --upstream-container-version 22.08 \
    --extra-backend-cmake-arg=${BACKEND}:TRITON_BUILD_OPENVINO_VERSION=2022.2.0 \
    --extra-backend-cmake-arg=${BACKEND}:TRITON_SRC_DIR=${PROJ_DIR}/src \
    --extra-backend-cmake-arg=${BACKEND}:absl_DIR:PATH=${ABSL_DIR} \
    --extra-backend-cmake-arg=${BACKEND}:TRITON_OPENVINO_LIB_PATHS=${LIBPT_DIR}/lib \
    --extra-backend-cmake-arg="${BACKEND}:TRITON_OPENVINO_INCLUDE_PATHS=${LIBPT_DIR}/include" \
    --cmake-dir ${PROJ_DIR} \
    --build-dir ${PROJ_DIR}/tritonbuild/backends \
    --install-dir ${PROJ_DIR}/tritonbuild/install

    mkdir -p ${PROJ_DIR}/tritonbuild/install/backends/openvino_2022_2
    cp ${PROJ_DIR}/tritonbuild/backends/openvino_2022_2/build/libtriton_openvino_2022_2.so \
      ${PROJ_DIR}/tritonbuild/install/backends/openvino_2022_2/

    cp ${LIBPT_DIR}/lib/libopenvino.so \
       ${LIBPT_DIR}/lib/libopenvino.so.2230 \
       ${LIBPT_DIR}/lib/libopenvino_arm_cpu_plugin.so \
       ${LIBPT_DIR}/lib/plugins.xml \
       ${PROJ_DIR}/tritonbuild/install/backends/openvino_2022_2/

    # cp ${LIBPT_DIR}/lib/libinference_engine.so \
    #    ${LIBPT_DIR}/lib/libiomp5.so \
    #    ${LIBPT_DIR}/lib/libinference_engine_transformations.so \
    #    ${LIBPT_DIR}/lib/libngraph.so \
    #    ${PROJ_DIR}/tritonbuild/install/backends/openvino_2021_4/
}

function build_openvino_backend() {
  # LIBPT_DIR=${PROJ_DIR}/tritonbuild/third_party/libopenvino_v22.08_2021.4
  LIBPT_DIR=${PROJ_DIR}/tritonbuild/third_party/libopenvino_v22.08_2022.2
  BACKEND=openvino
  python3 ./build.py -v \
    --backend=${BACKEND} \
    --no-container-build --no-core-build \
    --enable-gpu --enable-stats \
    --version 2.27.0 --container-version 22.08 --upstream-container-version 22.08 \
    --extra-backend-cmake-arg=${BACKEND}:TRITON_BUILD_OPENVINO_VERSION=2022.2.0 \
    --extra-backend-cmake-arg=${BACKEND}:TRITON_SRC_DIR=${PROJ_DIR}/src \
    --extra-backend-cmake-arg=${BACKEND}:absl_DIR:PATH=${ABSL_DIR} \
    --extra-backend-cmake-arg=${BACKEND}:TRITON_OPENVINO_LIB_PATHS=${LIBPT_DIR}/lib \
    --extra-backend-cmake-arg="${BACKEND}:TRITON_OPENVINO_INCLUDE_PATHS=${LIBPT_DIR}/include" \
    --cmake-dir ${PROJ_DIR} \
    --build-dir ${PROJ_DIR}/tritonbuild/backends \
    --install-dir ${PROJ_DIR}/tritonbuild/install

    mkdir -p ${PROJ_DIR}/tritonbuild/install/backends/openvino_2022_2
    cp ${PROJ_DIR}/tritonbuild/backends/openvino_2022_2/build/libtriton_openvino_2022_2.so \
      ${PROJ_DIR}/tritonbuild/install/backends/openvino_2022_2/

    cp ${LIBPT_DIR}/lib/libopenvino.so \
       ${LIBPT_DIR}/lib/libiomp5.so \
       ${LIBPT_DIR}/lib/libopenvino_intel_cpu_plugin.so \
       ${PROJ_DIR}/tritonbuild/install/backends/openvino_2022_2/

    pushd ${LIBPT_DIR}/lib
    cp plugins.xml \
       libopenvino_auto_plugin.so  \
       libopenvino_intel_gna_plugin.so   \
       libopenvino_hetero_plugin.so  \
       libopenvino_intel_myriad_plugin.so \
       libopenvino_auto_batch_plugin.so \
       libopenvino_ir_frontend.so  \
       ${PROJ_DIR}/tritonbuild/install/backends/openvino_2022_2/
    popd

    # cp ${LIBPT_DIR}/lib/libinference_engine.so \
    #    ${LIBPT_DIR}/lib/libiomp5.so \
    #    ${LIBPT_DIR}/lib/libinference_engine_transformations.so \
    #    ${LIBPT_DIR}/lib/libngraph.so \
    #    ${PROJ_DIR}/tritonbuild/install/backends/openvino_2021_4/
}

function build_enflame_backend() {
  LIBENFLAME_DIR=${PROJ_DIR}/tritonbuild/third_party/libenflame_v0.1/

  name=enflame_backend
  src_dir="${PROJ_DIR}/src/backends/$name"
  build_dir="${PROJ_DIR}/tritonbuild/backends/${name}/build"
  install_dir="${PROJ_DIR}/tritonbuild/backends/${name}/install"
  output_dir="${PROJ_DIR}/tritonbuild/install/backends"
  if [ ! -d ${build_dir} ]; then
    mkdir -p ${build_dir}
  fi
  rm -fr ${build_dir}/*
  pushd ${build_dir}
  cmake ${src_dir}                                        \
    -DCMAKE_INSTALL_PREFIX:PATH=${install_dir}            \
    -DPROJ_ROOT_DIR=${PROJ_DIR}                           \
    -DENFLAME_LIB_PATHS=${LIBENFLAME_DIR}/lib             \
    -DENFLAME_INCLUDE_PATHS=${LIBENFLAME_DIR}/include &&  \
    make -j $(nproc) && make install
  cp -fr ${install_dir}/backends/${name} ${output_dir}/
  popd
}

function build_cambricon_backend() {
  LIBCAMBRICON_DIR=${PROJ_DIR}/tritonbuild/third_party/libcambricon_v0.1/

  name=cambricon_backend
  src_dir="${PROJ_DIR}/src/backends/$name"
  build_dir="${PROJ_DIR}/tritonbuild/backends/${name}/build"
  install_dir="${PROJ_DIR}/tritonbuild/backends/${name}/install"
  output_dir="${PROJ_DIR}/tritonbuild/install/backends"
  if [ ! -d ${build_dir} ]; then
    mkdir -p ${build_dir}
  fi

  rm -fr ${build_dir}/*
  pushd ${build_dir}
  cmake ${src_dir}                                        \
    -DCMAKE_INSTALL_PREFIX:PATH=${install_dir}            \
    -DPROJ_ROOT_DIR=${PROJ_DIR}                           \
    -DCAMBRICON_LIB_PATHS=${LIBCAMBRICON_DIR}/lib             \
    -DCAMBRICON_INCLUDE_PATHS=${LIBCAMBRICON_DIR}/include &&  \
    make -j $(nproc) && make install

  mkdir -p ${PROJ_DIR}/tritonbuild/install/backends/cambricon
  cp ${PROJ_DIR}/tritonbuild/backends/cambricon_backend/build/libtriton_cambricon.so \
    ${PROJ_DIR}/tritonbuild/install/backends/cambricon/
  popd
}

function build_huawei_backend() {
  LIBHUAWEI_DIR=${PROJ_DIR}/tritonbuild/third_party/libhuawei_v0.1/

  name=huawei_backend
  src_dir="${PROJ_DIR}/src/backends/$name"
  build_dir="${PROJ_DIR}/tritonbuild/backends/${name}/build"
  install_dir="${PROJ_DIR}/tritonbuild/backends/${name}/install"
  output_dir="${PROJ_DIR}/tritonbuild/install/backends"
  if [ ! -d ${build_dir} ]; then
    mkdir -p ${build_dir}
  fi

  rm -fr ${build_dir}/*
  pushd ${build_dir}
  cmake ${src_dir}                                        \
    -DCMAKE_INSTALL_PREFIX:PATH=${install_dir}            \
    -DPROJ_ROOT_DIR=${PROJ_DIR}                           \
    -DHUAWEI_INFERENCE_DIR=${LIBHUAWEI_DIR}               \
    -DHUAWEI_LIB_PATHS=${LIBHUAWEI_DIR}/lib             \
    -DHUAWEI_INCLUDE_PATHS=${LIBHUAWEI_DIR}/include &&  \
    make -j $(nproc) && make install

  mkdir -p ${PROJ_DIR}/tritonbuild/install/backends/huawei
  cp ${PROJ_DIR}/tritonbuild/backends/huawei_backend/build/libtriton_huawei.so \
    ${PROJ_DIR}/tritonbuild/install/backends/huawei/
  popd
}

function build_server() {
  # rm -fr tritonbuild/tritonserver/build/triton-server/_deps/repo-core*

  # If update common or core library, need to uncommont follow
  rm -fr tritonbuild/tritonserver/build/_deps/repo-core-build

  #VERBOSE="-v"
  VERBOSE=""

  python3 ./build.py ${VERBOSE} \
    --no-container-build --enable-gpu \
    --version 2.27.0 --container-version 22.08 --upstream-container-version 22.08 \
    --backend ensemble \
    --enable-logging --enable-stats --enable-metrics --enable-gpu-metrics --enable-cpu-metrics \
    --enable-tracing --enable-nvtx \
    --endpoint grpc --endpoint http \
    --override-core-cmake-arg="TRITON_ENABLE_TENSORRT=ON" \
    --extra-core-cmake-arg="TRITON_SRC_DIR=${PROJ_DIR}/src" \
    --cmake-dir ${PROJ_DIR} \
    --build-dir ${PROJ_DIR}/tritonbuild \
    --install-dir ${PROJ_DIR}/tritonbuild/install
}

function build_server_cpu() {
  # rm -fr tritonbuild/tritonserver/build/triton-server/_deps/repo-core*
  # rm -fr tritonbuild/tritonserver/build/_deps/repo-core-build
  #    --extra-core-cmake-arg="LICENSE_STRATEGY:STRING=LICENSE_HASP" \

  python3 ./build.py -v \
    --no-container-build \
    --version 2.27.0 --container-version 22.08 --upstream-container-version 22.08 \
    --backend ensemble \
    --enable-logging --enable-stats --enable-metrics \
    --enable-tracing \
    --endpoint grpc --endpoint http \
    --override-core-cmake-arg="TRITON_ENABLE_TENSORRT=OFF" \
    --override-core-cmake-arg="ENABLE_RAPIDJSON_SIMD=OFF" \
    --extra-core-cmake-arg="LICENSE_STRATEGY:STRING=LICENSE_HASP" \
    --extra-core-cmake-arg="LICENSE_HASP_VCODE:STRING=DEMOMA" \
    --extra-core-cmake-arg="TRITON_SRC_DIR=${PROJ_DIR}/src" \
    --cmake-dir ${PROJ_DIR} \
    --build-dir ${PROJ_DIR}/tritonbuild \
    --install-dir ${PROJ_DIR}/tritonbuild/install
}

function build_backend_example() {
  name=$1
  src_dir="${PROJ_DIR}/src/backend/examples/backends/$name"
  build_dir="${PROJ_DIR}/tritonbuild/backends/${name}/build"
  install_dir="${PROJ_DIR}/tritonbuild/backends/${name}/install"
  output_dir="${PROJ_DIR}/tritonbuild/install/backends"
  if [ ! -d ${build_dir} ]; then
    mkdir -p ${build_dir}
  fi

  pushd ${build_dir}
  cmake ${src_dir}                              \
    -DCMAKE_PREFIX_PATH=${ABSL_DIR}             \
    -DCMAKE_INSTALL_PREFIX:PATH=${install_dir}  \
    -DTRITON_COMMON_REPO_TAG=r22.08             \
    -DTRITON_CORE_REPO_TAG=r22.08               \
    -DTRITON_BACKEND_REPO_TAG=r22.08 &&         \
    make -j $(nproc) && make install
  cp -fr ${install_dir}/backends/${name} ${output_dir}/
  popd
}

function build_dataelem_backend() {
  name=dataelem
  src_dir="${PROJ_DIR}/src/backends/${name}_backend"
  build_dir="${PROJ_DIR}/tritonbuild/backends/${name}/build"
  install_dir="${PROJ_DIR}/tritonbuild/backends/${name}/install"
  output_dir="${PROJ_DIR}/tritonbuild/install/backends"
  if [ ! -d ${build_dir} ]; then
    mkdir -p ${build_dir}
  fi

  if [ ! -d ${output_dir} ]; then
    mkdir -p ${output_dir}
  fi

  ENABLE_GPU=ON

  pushd ${build_dir}
  cmake ${src_dir}                              \
    -DCMAKE_INSTALL_PREFIX:PATH=${install_dir}  \
    -DPROJ_ROOT_DIR=${PROJ_DIR}                 \
    -DTRITON_ENABLE_GPU:BOOL=${ENABLE_GPU}      \
    -DTRITON_COMMON_REPO_TAG=r22.08             \
    -DTRITON_CORE_REPO_TAG=r22.08               \
    -DTRITON_BACKEND_REPO_TAG=r22.08 &&         \
    make -j $(nproc) && make install
  cp -fr ${install_dir}/backends/${name} ${output_dir}/
  popd
}


function build_dataelem_backend_cpu() {
  name=dataelem
  src_dir="${PROJ_DIR}/src/backends/${name}_backend"
  build_dir="${PROJ_DIR}/tritonbuild/backends/${name}/build"
  install_dir="${PROJ_DIR}/tritonbuild/backends/${name}/install"
  output_dir="${PROJ_DIR}/tritonbuild/install_cpu/backends"
  if [ ! -d ${build_dir} ]; then
    mkdir -p ${build_dir}
  fi

  if [ ! -d ${output_dir} ]; then
    mkdir -p ${output_dir}
  fi

  ENABLE_GPU=OFF

  pushd ${build_dir}
  cmake ${src_dir}                              \
    -DCMAKE_INSTALL_PREFIX:PATH=${install_dir}  \
    -DPROJ_ROOT_DIR=${PROJ_DIR}                 \
    -DTRITON_ENABLE_GPU:BOOL=${ENABLE_GPU}      \
    -DTRITON_COMMON_REPO_TAG=r22.08             \
    -DTRITON_CORE_REPO_TAG=r22.08               \
    -DTRITON_BACKEND_REPO_TAG=r22.08 &&         \
    make -j $(nproc) && make install
  cp -fr ${install_dir}/backends/${name} ${output_dir}/
  popd
}


function build_client() {
  src_dir="${PROJ_DIR}/src/client"
  build_dir="${PROJ_DIR}/tritonbuild/client/build"
  install_dir="${PROJ_DIR}/tritonbuild/client/install"
  output_dir="${PROJ_DIR}/tritonbuild/install"
  if [ ! -d ${build_dir} ]; then
    mkdir -p ${build_dir}
  fi

  pushd ${build_dir}
  cmake ${src_dir} \
    -DCMAKE_PREFIX_PATH=${ABSL_DIR}            \
    -DCMAKE_INSTALL_PREFIX:PATH=${install_dir} \
    -DTRITON_ENABLE_GPU:BOOL=ON                \
    -DTRITON_COMMON_REPO_TAG=r22.08            \
    -DTRITON_CORE_REPO_TAG=r22.08              \
    -DTRITON_BACKEND_REPO_TAG=r22.08           \
    -DTRITON_ENABLE_CC_HTTP=ON                 \
    -DTRITON_ENABLE_CC_GRPC=ON                 \
    -DTRITON_ENABLE_PERF_ANALYZER=ON           \
    -DTRITON_ENABLE_PERF_ANALYZER_C_API=ON     \
    -DTRITON_ENABLE_PERF_ANALYZER_TFS=OFF      \
    -DTRITON_ENABLE_PERF_ANALYZER_TS=OFF       \
    -DTRITON_SRC_DIR=${PROJ_DIR}/src           \
    -DTRITON_ENABLE_EXAMPLES=ON                \
    -DTRITON_ENABLE_TESTS=ON  &&               \
    make -j $(nproc)

  mkdir -p ${output_dir}/client/ || true
  cp -fr ${install_dir}/* ${output_dir}/client/
  popd
}

function build_dataelem_python_backend() {
  name=dataelem_python
  src_dir="${PROJ_DIR}/src/backends/${name}_backend"
  build_dir="${PROJ_DIR}/tritonbuild/backends/${name}/build"
  install_dir="${PROJ_DIR}/tritonbuild/backends/${name}/install"
  output_dir="${PROJ_DIR}/tritonbuild/install/backends"
  if [ ! -d ${build_dir} ]; then
    mkdir -p ${build_dir}
  fi

  pushd ${build_dir}
  cmake ${src_dir}                               \
    -DCMAKE_PREFIX_PATH=${ABSL_DIR}              \
    -DCMAKE_INSTALL_PREFIX:PATH=${install_dir}   \
    -DTRITON_COMMON_REPO_TAG=r22.08              \
    -DTRITON_CORE_REPO_TAG=r22.08                \
    -DTRITON_BACKEND_REPO_TAG=r22.08             \
    -DTRITON_SRC_DIR=${PROJ_DIR}/src             \
    -DPROJ_ROOT_DIR=${PROJ_DIR} &&               \
    make -j $(nproc) && make install

  cp -fr ${install_dir}/backends/${name} ${output_dir}/
  popd
}

function build_test() {
  src_dir="${PROJ_DIR}/src/tests"
  build_dir="${PROJ_DIR}/tritonbuild/tests/build"
  install_dir="${PROJ_DIR}/tritonbuild/tests/install"
  output_dir="${PROJ_DIR}/tritonbuild/install"
  if [ ! -d ${build_dir} ]; then
    mkdir -p ${build_dir}
  fi

  pushd ${build_dir}
  cmake ${src_dir}                          \
    -DTRITON_ENABLE_GPU:BOOL=ON             \
    -DPROJ_ROOT_DIR=${PROJ_DIR} &&          \
    make -j $(nproc)

  # if [ -f ./ut_all ]; then
  #   ./ut_all
  # fi
  popd
}


function extra_deps() {
  apt update && apt install -y libarchive-dev patchelf libgl1
}



opt="$1"
case $opt in
  "full")
    build_server && \
      build_dataelem_backend && \
      build_python_backend && \
      build_tf_backend tensorflow2 && \
      build_tensorrt_backend && \
      build_onnx_backend && \
      build_pt_backend
    ;;
  "server")
    build_server
    ;;
  "backends")
    # build_dataelem_backend && \
    #   build_python_backend && \
    #   build_tf_backend tensorflow2 && \
    #   build_tensorrt_backend && \
    #   build_onnx_backend && \
    #   build_pt_backend
    build_python_backend
    ;;
  *)
    echo "error parameters"
    ;;
esac

if [[ $? -eq 0 ]]; then
  echo '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
  echo "succ to build $opt"
else
  echo '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'
  echo "failed to build $opt"
fi
