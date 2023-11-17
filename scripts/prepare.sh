#!/bin/bash


prepare_zh_env(){
  apt update
  # yum -y install kde-l10n-Chinese && yum -y reinstall glibc-common
  apt-get -y install language-pack-zh-hans
  localedef -c -f UTF-8 -i zh_CN zh_CN.utf8
  export LC_ALL=zh_CN.utf8
}

prepare_cv_syslib() {
  apt install -y libsm6 libxext6 libxrender-dev
}

prepare_system_lib() {

    apt install -y nasm
    apt install -y zlib1g-dev
    apt install -y rapidjson-dev
    apt install -y libssl-dev
    apt install -y libboost1.71-dev
    apt install -y libre2-dev
    apt install -y libb64-dev
    apt install -y libarchive-dev
}    

prepare_python_be() {
    apt install -y python3.8 libpython3.8-dev python3-pip
    local repo="https://mirrors.aliyun.com/pypi/simple"
    pip3 install --upgrade wheel setuptools -i repo
    pip3 install --upgrade numpy -i $repo
}


prepre_dcgm() {
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
  wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb
  dpkg -i cuda-keyring_1.0-1_all.deb
  apt-get update
  apt-get install -y datacenter-gpu-manager
  rm cuda-keyring_1.0-1_all.deb
}


prepare_tf_libs() {
    ver="$1"
    TRITON_TENSORFLOW_VERSION=$ver
    TRITON_TENSORFLOW_DOCKER_IMAGE="nvcr.io/nvidia/tensorflow:22.08-tf${ver}-py3"
    TRITON_TENSORFLOW_CC_LIBNAME="libtensorflow_cc.so"
    TRITON_TENSORFLOW_FW_LIBNAME="libtensorflow_framework.so"

    if [[ $ver == "1" ]]; then
      TRITON_TENSORFLOW_PYTHON_PATH="/usr/local/lib/python3.8/dist-packages/tensorflow_core"
    else 
      TRITON_TENSORFLOW_PYTHON_PATH="/usr/local/lib/python3.8/dist-packages/tensorflow"
    fi

    OUT_DIR="./tritonbuild/third_party/libtf${ver}_v22.08"
    LIB_DIR="${OUT_DIR}/lib"
    mkdir -p ${LIB_DIR} || echo "ignore..." || true
    mkdir -p ${LIB_DIR}/tf_backend_deps || echo "ignore..." || true

    LIBS_ARCH=x86_64
    # docker pull ${TRITON_TENSORFLOW_DOCKER_IMAGE}
    docker rm tensorflow_backend_tflib || echo "error ignored..." || true

    # docker create --name tensorflow_backend_tflib ${TRITON_TENSORFLOW_DOCKER_IMAGE}
    docker run -it -d --name tensorflow_backend_tflib ${TRITON_TENSORFLOW_DOCKER_IMAGE}

    docker cp -L tensorflow_backend_tflib:/usr/local/lib/tensorflow/${TRITON_TENSORFLOW_CC_LIBNAME}.${TRITON_TENSORFLOW_VERSION} \
      ${LIB_DIR}/${TRITON_TENSORFLOW_CC_LIBNAME}.${TRITON_TENSORFLOW_VERSION}

    docker cp tensorflow_backend_tflib:${TRITON_TENSORFLOW_PYTHON_PATH}/${TRITON_TENSORFLOW_FW_LIBNAME}.${TRITON_TENSORFLOW_VERSION} \
      ${LIB_DIR}/${TRITON_TENSORFLOW_FW_LIBNAME}.${TRITON_TENSORFLOW_VERSION}

    docker exec tensorflow_backend_tflib sh -c  "tar -cf - /usr/lib/${LIBS_ARCH}-linux-gnu/libnccl.so*" | \
      tar --strip-components=3 -xf - -C ${LIB_DIR}/tf_backend_deps

    docker cp tensorflow_backend_tflib:${TRITON_TENSORFLOW_PYTHON_PATH}/include ${OUT_DIR}/
    docker cp tensorflow_backend_tflib:/opt/tensorflow/tensorflow-source/LICENSE ${OUT_DIR}/LICENSE.tensorflow

    docker stop tensorflow_backend_tflib
    docker rm tensorflow_backend_tflib
}

function preinstall() {
  # wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
  #   gpg --dearmor - |  \
  #   tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null

  # apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
  # apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
  # apt-get update
  # apt-get install -y --no-install-recommends --option Acquire::HTTP::Proxy=http://192.168.106.8:1081 \
  #   cmake-data=3.21.1-0kitware1ubuntu20.04.1 cmake=3.21.1-0kitware1ubuntu20.04.1 
  wget http://192.168.106.8:8100/deps/cmake-3.23.1-linux-x86_64.tar.gz
  tar zxf cmake-3.23.1-linux-x86_64.tar.gz --strip-components 1 -C /usr/local
  rm cmake-3.23.1-linux-x86_64.tar.gz
}

prepare_dali_be() {
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

}

prepare_pt_be() {
    TRITON_PYTORCH_DOCKER_IMAGE="nvcr.io/nvidia/pytorch:22.08-py3"

    OUT_DIR="$(pwd)/tritonbuild/third_party/libtorch_v22.08"
    LIB_DIR="${OUT_DIR}/lib"
    BIN_DIR="${OUT_DIR}/bin"
    INC_DIR="${OUT_DIR}/include"

    mkdir -p ${LIB_DIR} || echo "ignore..." || true
    mkdir -p ${BIN_DIR} || echo "ignore..." || true
    mkdir -p ${INC_DIR} || echo "ignore..." || true

    # ${CMAKE_COMMAND} -E make_directory "include/torchvision"
    # docker pull ${TRITON_PYTORCH_DOCKER_IMAGE}
    LIBS_ARCH=x86_64
    TRITON_PYTORCH_ENABLE_TORCHTRT=ON

    CONDA_LIBS=(
      "libmkl_core.so"
      "libmkl_gnu_thread.so"
      "libmkl_intel_lp64.so"
      "libmkl_intel_thread.so"
      "libmkl_def.so"
      "libmkl_vml_def.so"
      "libmkl_rt.so"
      "libmkl_avx2.so"
      "libmkl_avx512.so"
      "libmkl_sequential.so"
      "libomp.so"
    )

    docker rm pytorch_backend_ptlib || echo "error ignored..." || true
    docker create --name pytorch_backend_ptlib ${TRITON_PYTORCH_DOCKER_IMAGE}

    pushd ${LIB_DIR}

    for _lib in ${CONDA_LIBS[@]}; do
      docker cp -L pytorch_backend_ptlib:/opt/conda/lib/${_lib} ${_lib}
    done

    docker cp pytorch_backend_ptlib:/opt/conda/lib/python3.8/site-packages/torch/lib/libc10.so libc10.so
    docker cp pytorch_backend_ptlib:/opt/conda/lib/python3.8/site-packages/torch/lib/libc10_cuda.so libc10_cuda.so
    docker cp pytorch_backend_ptlib:/opt/conda/lib/python3.8/site-packages/torch/lib/libtorch.so libtorch.so
    docker cp pytorch_backend_ptlib:/opt/conda/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so libtorch_cpu.so
    docker cp pytorch_backend_ptlib:/opt/conda/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so libtorch_cuda.so
    docker cp pytorch_backend_ptlib:/opt/conda/lib/python3.8/site-packages/torch/lib/libtorch_global_deps.so libtorch_global_deps.so
    docker cp pytorch_backend_ptlib:/opt/pytorch/vision/build/libtorchvision.so libtorchvision.so

    if [ ${TRITON_PYTORCH_ENABLE_TORCHTRT} = 'ON' ]; then 
      docker cp pytorch_backend_ptlib:/opt/conda/lib/python3.8/site-packages/torch_tensorrt/lib/libtorchtrt_runtime.so libtorchtrt_runtime.so; 
    fi
    
    docker cp pytorch_backend_ptlib:/opt/conda/lib/python3.8/site-packages/torch_tensorrt/bin/torchtrtc ${BIN_DIR}/torchtrtc || echo "error ignored..." || true
    docker cp pytorch_backend_ptlib:/opt/pytorch/pytorch/LICENSE ${OUT_DIR}/LICENSE.pytorch
    docker cp pytorch_backend_ptlib:/opt/conda/lib/python3.8/site-packages/torch/include ${OUT_DIR}/include/torch
    docker cp pytorch_backend_ptlib:/opt/pytorch/pytorch/torch/csrc/jit/codegen ${OUT_DIR}/include/torch/torch/csrc/jit/.
    docker cp pytorch_backend_ptlib:/opt/pytorch/vision/torchvision/csrc ${OUT_DIR}/include/torchvision/torchvision

    docker cp pytorch_backend_ptlib:/usr/lib/${LIBS_ARCH}-linux-gnu/libopencv_videoio.so.3.4.11 libopencv_videoio.so
    docker cp pytorch_backend_ptlib:/usr/lib/${LIBS_ARCH}-linux-gnu/libopencv_highgui.so.3.4.11 libopencv_highgui.so
    docker cp pytorch_backend_ptlib:/usr/lib/${LIBS_ARCH}-linux-gnu/libopencv_video.so.3.4.11 libopencv_video.so
    docker cp pytorch_backend_ptlib:/usr/lib/${LIBS_ARCH}-linux-gnu/libopencv_imgcodecs.so.3.4.11 libopencv_imgcodecs.so
    docker cp pytorch_backend_ptlib:/usr/lib/${LIBS_ARCH}-linux-gnu/libopencv_imgproc.so.3.4.11 libopencv_imgproc.so
    docker cp pytorch_backend_ptlib:/usr/lib/${LIBS_ARCH}-linux-gnu/libopencv_core.so.3.4.11 libopencv_core.so
    docker cp pytorch_backend_ptlib:/usr/lib/${LIBS_ARCH}-linux-gnu/libpng16.so.16.37.0 libpng16.so
    docker cp pytorch_backend_ptlib:/usr/lib/${LIBS_ARCH}-linux-gnu/libjpeg.so.8.2.2 libjpeg.so

    # /bin/sh -c "if [ -f libmkl_def.so ]; then patchelf --add-needed libmkl_gnu_thread.so libmkl_def.so; fi"
    # /bin/sh -c "if [ -f libmkl_def.so ]; then patchelf --add-needed libmkl_core.so libmkl_def.so; fi"
    # /bin/sh -c "if [ -f libmkl_avx2.so ]; then patchelf --add-needed libmkl_gnu_thread.so libmkl_avx2.so; fi"
    # /bin/sh -c "if [ -f libmkl_avx2.so ]; then patchelf --add-needed libmkl_core.so libmkl_avx2.so; fi"
    # /bin/sh -c "if [ -f libmkl_avx512.so ]; then patchelf --add-needed libmkl_gnu_thread.so libmkl_avx512.so; fi"
    # /bin/sh -c "if [ -f libmkl_avx512.so ]; then patchelf --add-needed libmkl_core.so libmkl_avx512.so; fi"
    # /bin/sh -c "if [ -f libmkl_vml_def.so ]; then patchelf --add-needed libmkl_gnu_thread.so libmkl_vml_def.so; fi"
    # /bin/sh -c "if [ -f libmkl_vml_def.so ]; then patchelf --add-needed libmkl_intel_thread.so libmkl_vml_def.so; fi"
    # /bin/sh -c "if [ -f libmkl_vml_def.so ]; then patchelf --add-needed libmkl_core.so libmkl_vml_def.so; fi"
    # /bin/sh -c "if [ -f libmkl_intel_thread.so ]; then patchelf --add-needed libmkl_intel_lp64.so libmkl_intel_thread.so; fi"
    docker rm pytorch_backend_ptlib
}

fix_pt_lib() {
    OUT_DIR="$(pwd)/tritonbuild/third_party/libtorch_v22.08"
    LIB_DIR="${OUT_DIR}/lib"
    pushd ${LIB_DIR}

    /bin/sh -c "if [ -f libmkl_def.so ]; then patchelf --add-needed libmkl_gnu_thread.so libmkl_def.so; fi"
    /bin/sh -c "if [ -f libmkl_def.so ]; then patchelf --add-needed libmkl_core.so libmkl_def.so; fi"
    /bin/sh -c "if [ -f libmkl_avx2.so ]; then patchelf --add-needed libmkl_gnu_thread.so libmkl_avx2.so; fi"
    /bin/sh -c "if [ -f libmkl_avx2.so ]; then patchelf --add-needed libmkl_core.so libmkl_avx2.so; fi"
    /bin/sh -c "if [ -f libmkl_avx512.so ]; then patchelf --add-needed libmkl_gnu_thread.so libmkl_avx512.so; fi"
    /bin/sh -c "if [ -f libmkl_avx512.so ]; then patchelf --add-needed libmkl_core.so libmkl_avx512.so; fi"
    /bin/sh -c "if [ -f libmkl_vml_def.so ]; then patchelf --add-needed libmkl_gnu_thread.so libmkl_vml_def.so; fi"
    /bin/sh -c "if [ -f libmkl_vml_def.so ]; then patchelf --add-needed libmkl_intel_thread.so libmkl_vml_def.so; fi"
    /bin/sh -c "if [ -f libmkl_vml_def.so ]; then patchelf --add-needed libmkl_core.so libmkl_vml_def.so; fi"
    /bin/sh -c "if [ -f libmkl_intel_thread.so ]; then patchelf --add-needed libmkl_intel_lp64.so libmkl_intel_thread.so; fi"
}

fix_tf_lib() {
    OUT_DIR="$(pwd)/tritonbuild/third_party/libtf2_v22.08"
    LIB_DIR="${OUT_DIR}/lib"
    pushd ${LIB_DIR}
    ln -s libtensorflow_cc.so.2 libtensorflow_cc.so
    ln -s libtensorflow_framework.so.2 libtensorflow_framework.so
}

fix_tf1_lib() {
    OUT_DIR="$(pwd)/tritonbuild/third_party/libtf1_v22.08"
    LIB_DIR="${OUT_DIR}/lib"
    pushd ${LIB_DIR}
    ln -s libtensorflow_cc.so.1 libtensorflow_cc.so
    ln -s libtensorflow_framework.so.1 libtensorflow_framework.so
}

prepare_tf_libs_new() {
    ver="$1"
    TRITON_TENSORFLOW_VERSION=$ver
    TRITON_TENSORFLOW_CC_LIBNAME="libtensorflow_cc.so"
    TRITON_TENSORFLOW_FW_LIBNAME="libtensorflow_framework.so"

    if [[ $ver == "1" ]]; then
      CONT_NAME=tf1_dev
      TRITON_TENSORFLOW_PYTHON_PATH="/usr/local/lib/python3.8/dist-packages/tensorflow_core"
    else 
      CONT_NAME=hf_tf2_2208
      TRITON_TENSORFLOW_PYTHON_PATH="/usr/local/lib/python3.8/dist-packages/tensorflow"
    fi

    OUT_DIR="./tritonbuild/third_party/libtf${ver}_v22.08"
    LIB_DIR="${OUT_DIR}/lib"
    mkdir -p ${LIB_DIR} || echo "ignore..." || true
    mkdir -p ${LIB_DIR}/tf_backend_deps || echo "ignore..." || true

    LIBS_ARCH=x86_64

    docker cp -L ${CONT_NAME}:/usr/local/lib/tensorflow/${TRITON_TENSORFLOW_CC_LIBNAME}.${TRITON_TENSORFLOW_VERSION} \
      ${LIB_DIR}/${TRITON_TENSORFLOW_CC_LIBNAME}.${TRITON_TENSORFLOW_VERSION}
    docker cp ${CONT_NAME}:${TRITON_TENSORFLOW_PYTHON_PATH}/${TRITON_TENSORFLOW_FW_LIBNAME}.${TRITON_TENSORFLOW_VERSION} \
      ${LIB_DIR}/${TRITON_TENSORFLOW_FW_LIBNAME}.${TRITON_TENSORFLOW_VERSION}

    docker exec ${CONT_NAME} sh -c  "tar -cf - /usr/lib/${LIBS_ARCH}-linux-gnu/libnccl.so*" | \
      tar --strip-components=3 -xf - -C ${LIB_DIR}/tf_backend_deps

    docker cp ${CONT_NAME}:${TRITON_TENSORFLOW_PYTHON_PATH}/include ${OUT_DIR}/
    docker cp ${CONT_NAME}:/opt/tensorflow/tensorflow-source/LICENSE ${OUT_DIR}/LICENSE.tensorflow
  
    # patch for missing headers
    TF_SRC_PATH="/opt/tensorflow/tensorflow-source/tensorflow"
    OUT_GPU_INC_PATH="${OUT_DIR}/include/tensorflow/core/common_runtime/gpu/"
    if [[ ! -d ${OUT_GPU_INC_PATH} ]]; then
       mkdir -p ${OUT_GPU_INC_PATH}
    fi
    
    if [[ $ver == "1" ]]; then
      docker cp ${CONT_NAME}:${TF_SRC_PATH}/core/common_runtime/gpu/gpu_id_utils.h ${OUT_GPU_INC_PATH}
      docker cp ${CONT_NAME}:${TF_SRC_PATH}/core/common_runtime/gpu/gpu_mem_allocator.h ${OUT_GPU_INC_PATH}
      docker cp ${CONT_NAME}:/opt/tensorflow/tensorflow-source/tensorflow/c ${OUT_DIR}/include/tensorflow/c
    fi

    docker cp ${CONT_NAME}:${TF_SRC_PATH}/core/common_runtime/gpu/gpu_process_state.h ${OUT_GPU_INC_PATH}

    OUT_SAVED_MODEL_INC_PATH="${OUT_DIR}/include/tensorflow/cc/saved_model/"
    if [[ ! -d ${OUT_SAVED_MODEL_INC_PATH} ]]; then
       mkdir -p ${OUT_SAVED_MODEL_INC_PATH}
    fi
  
    docker cp ${CONT_NAME}:${TF_SRC_PATH}/cc/saved_model/tag_constants.h ${OUT_SAVED_MODEL_INC_PATH}
    docker cp ${CONT_NAME}:${TF_SRC_PATH}/cc/saved_model/loader.h ${OUT_SAVED_MODEL_INC_PATH}
    
}

function patch_pip() {
  # patch for install transformers
  ln -s /usr/local/bin/pip3.8 /usr/bin/
}


function clean() {
   echo "clean" 
   apt-get clean &&  rm -rf /var/lib/apt/lists/* && rm -rf /root/.cache/pip
}


function commit_dev_image() {
    build_image="dataelement/bisheng-rt-base:0.0.1"
    docker rmi ${build_image}
    LOCAL_HOME=$HOME
    docker commit -a "hanfeng@dataelem.com" -m "commit bisheng-rt base dev image" bisheng_rt_v005_dev ${build_image}
}


# prepare_python_be
# prepare_tf_libs 1
# prepare_tf_libs 2

# prepare_dali_be
# prepare_pt_be

# prepare_system_lib

# prepare_tf_libs_new 1
# fix_tf1_lib

# prepare_tf_libs_new 2
# fix_tf_lib


## dependences for bisheng-rt

# prepare_zh_env
# prepare_system_lib
# prepare_cv_syslib
# prepare_python_be
# preinstall
# prepre_dcgm
# clean

# commit_dev_image

# upload_tf_pkg
# install_tf_pkg