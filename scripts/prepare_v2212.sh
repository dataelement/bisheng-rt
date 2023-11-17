#!/bin/bash

prepare_host_env() {
  # manually get https://github.com/NixOS/patchelf/releases
  tar zxf patchelf-0.18.0-x86_64.tar.gz --strip-components 1 -C $HOME/.local
}


create_pt_compile_env() {
  IMAGE="nvcr.io/nvidia/pytorch:22.12-py3"
  docker run --gpus=all --net=host -itd  --shm-size=10G --name pt_2212_dev $IMAGE bash
}


prepare_pt_be() {
    TRITON_PYTORCH_DOCKER_IMAGE="nvcr.io/nvidia/pytorch:22.12-py3"
    OUT_DIR="$(pwd)/tritonbuild/third_party/libtorch_v22.12"
    rm -fr ${OUT_DIR}

    LIB_DIR="${OUT_DIR}/lib"
    BIN_DIR="${OUT_DIR}/bin"
    INC_DIR="${OUT_DIR}/include"

    mkdir -p ${LIB_DIR} || echo "ignore..." || true
    mkdir -p ${BIN_DIR} || echo "ignore..." || true
    mkdir -p ${INC_DIR} || echo "ignore..." || true

    LIBS_ARCH=x86_64
    TRITON_PYTORCH_ENABLE_TORCHTRT=ON

    if [ ${LIBS_ARCH} = 'aarch64' ]; then
      LIBTORCH_LIBS=("libopenblas.so.0")
    else
      LIBTORCH_LIBS=(
        "libmkl_avx2.so.1"
        "libmkl_avx512.so.1"
        "libmkl_core.so.1"
        "libmkl_def.so.1"
        "libmkl_gnu_thread.so.1"
        "libmkl_intel_lp64.so.1"
        "libmkl_intel_thread.so.1"
        "libmkl_rt.so.1"
        "libmkl_sequential.so.1"
        "libmkl_vml_def.so.1")
    fi

    OPENCV_LIBS=(
      "libopencv_video.so"
      "libopencv_videoio.so"
      "libopencv_highgui.so"
      "libopencv_imgcodecs.so"
      "libopencv_imgproc.so"
      "libopencv_core.so" 
      "libopencv_calib3d.so"
      "libopencv_flann.so"
      "libopencv_features2d.so"
      "libpng16.so"
      "libjpeg.so")

    pushd ${LIB_DIR}

    docker rm pytorch_backend_ptlib || echo "error ignored..." || true
    docker create --name pytorch_backend_ptlib ${TRITON_PYTORCH_DOCKER_IMAGE}

    for i in ${LIBTORCH_LIBS[@]}; do 
      echo copying $i && docker cp -L pytorch_backend_ptlib:/usr/local/lib/$i $i
    done
    
    docker cp pytorch_backend_ptlib:/usr/local/lib/python3.8/dist-packages/torch/lib/libc10.so libc10.so
    docker cp pytorch_backend_ptlib:/usr/local/lib/python3.8/dist-packages/torch/lib/libc10_cuda.so libc10_cuda.so
    docker cp pytorch_backend_ptlib:/usr/local/lib/python3.8/dist-packages/torch/lib/libtorch.so libtorch.so
    docker cp pytorch_backend_ptlib:/usr/local/lib/python3.8/dist-packages/torch/lib/libtorch_cpu.so libtorch_cpu.so
    docker cp pytorch_backend_ptlib:/usr/local/lib/python3.8/dist-packages/torch/lib/libtorch_cuda.so libtorch_cuda.so
    docker cp pytorch_backend_ptlib:/usr/local/lib/python3.8/dist-packages/torch/lib/libtorch_global_deps.so libtorch_global_deps.so
    docker cp pytorch_backend_ptlib:/usr/local/lib/python3.8/dist-packages/torch/lib/libcaffe2_nvrtc.so libcaffe2_nvrtc.so
    docker cp pytorch_backend_ptlib:/usr/local/lib/libtorchvision.so libtorchvision.so
    /bin/sh -c "if [ ${TRITON_PYTORCH_ENABLE_TORCHTRT} = 'ON' ]; then docker cp pytorch_backend_ptlib:/usr/local/lib/python3.8/dist-packages/torch_tensorrt/lib/libtorchtrt_runtime.so libtorchtrt_runtime.so; fi"

    docker cp -L pytorch_backend_ptlib:/usr/local/lib/libopencv_videoio.so libopencv_videoio.so
    docker cp -L pytorch_backend_ptlib:/usr/local/lib/libopencv_highgui.so libopencv_highgui.so
    docker cp -L pytorch_backend_ptlib:/usr/local/lib/libopencv_video.so libopencv_video.so
    docker cp -L pytorch_backend_ptlib:/usr/local/lib/libopencv_imgcodecs.so libopencv_imgcodecs.so
    docker cp -L pytorch_backend_ptlib:/usr/local/lib/libopencv_imgproc.so libopencv_imgproc.so
    docker cp -L pytorch_backend_ptlib:/usr/local/lib/libopencv_core.so libopencv_core.so
    docker cp -L pytorch_backend_ptlib:/usr/local/lib/libopencv_calib3d.so libopencv_calib3d.so
    docker cp -L pytorch_backend_ptlib:/usr/local/lib/libopencv_features2d.so libopencv_features2d.so
    docker cp -L pytorch_backend_ptlib:/usr/local/lib/libopencv_flann.so libopencv_flann.so
    docker cp pytorch_backend_ptlib:/usr/lib/${LIBS_ARCH}-linux-gnu/libpng16.so.16.37.0 libpng16.so
    docker cp pytorch_backend_ptlib:/usr/lib/${LIBS_ARCH}-linux-gnu/libjpeg.so.8.2.2 libjpeg.so
    /bin/sh -c "if [ -f libmkl_def.so.1 ]; then patchelf --add-needed libmkl_gnu_thread.so.1 libmkl_def.so.1; fi"
    /bin/sh -c "if [ -f libmkl_def.so.1 ]; then patchelf --add-needed libmkl_core.so.1 libmkl_def.so.1; fi"
    /bin/sh -c "if [ -f libmkl_avx2.so.1 ]; then patchelf --add-needed libmkl_gnu_thread.so.1 libmkl_avx2.so.1; fi"
    /bin/sh -c "if [ -f libmkl_avx2.so.1 ]; then patchelf --add-needed libmkl_core.so.1 libmkl_avx2.so.1; fi"
    /bin/sh -c "if [ -f libmkl_avx512.so.1 ]; then patchelf --add-needed libmkl_gnu_thread.so.1 libmkl_avx512.so.1; fi"
    /bin/sh -c "if [ -f libmkl_avx512.so.1 ]; then patchelf --add-needed libmkl_core.so.1 libmkl_avx512.so.1; fi"
    /bin/sh -c "if [ -f libmkl_vml_def.so.1 ]; then patchelf --add-needed libmkl_gnu_thread.so.1 libmkl_vml_def.so.1; fi"
    /bin/sh -c "if [ -f libmkl_vml_def.so.1 ]; then patchelf --add-needed libmkl_intel_thread.so.1 libmkl_vml_def.so.1; fi"
    /bin/sh -c "if [ -f libmkl_vml_def.so.1 ]; then patchelf --add-needed libmkl_core.so.1 libmkl_vml_def.so.1; fi"
    /bin/sh -c "if [ -f libmkl_intel_thread.so.1 ]; then patchelf --add-needed libmkl_intel_lp64.so.1 libmkl_intel_thread.so.1; fi"
    
    pushd ${OUT_DIR}
    docker cp pytorch_backend_ptlib:/usr/local/lib/python3.8/dist-packages/torch/include include/torch
    docker cp pytorch_backend_ptlib:/opt/pytorch/pytorch/torch/csrc/jit/codegen include/torch/torch/csrc/jit/.
    docker cp pytorch_backend_ptlib:/opt/pytorch/vision/torchvision/csrc include/torchvision/torchvision

    docker cp pytorch_backend_ptlib:/opt/pytorch/pytorch/LICENSE LICENSE.pytorch
    docker cp pytorch_backend_ptlib:/usr/local/lib/python3.8/dist-packages/torch_tensorrt/bin/torchtrtc bin/torchtrtc || echo "error ignored..." || true

    docker rm pytorch_backend_ptlib
    echo "Extracting pytorch and torchvision libraries and includes from ${TRITON_PYTORCH_DOCKER_IMAGE}"

    # patch origin

    PT_LIBS=(
      "libc10.so"
      "libc10_cuda.so"
      "libtorch.so"
      "libtorch_cpu.so"
      "libtorch_cuda.so"
      "libtorch_global_deps.so"
      "libtorchtrt_runtime.so"
      "libtorchvision.so"
    )

    pushd ${LIB_DIR}
    for plib in ${PT_LIBS[@]}; do
      patchelf --set-rpath \$ORIGIN ${plib}
    done

    for plib in ${LIBTORCH_LIBS[@]}; do
      patchelf --set-rpath \$ORIGIN ${plib}
    done

    for plib in ${OPENCV_LIBS[@]}; do
      patchelf --set-rpath \$ORIGIN ${plib}
    done

    echo "Succ to patch pytorch and torchvision libraries"
}


create_tf_compile_env() {
  ver="$1"
  IMAGE="nvcr.io/nvidia/tensorflow:22.12-tf${ver}-py3"
  # docker run --gpus=all --net=host -itd  --shm-size=10G --name tf${ver}_2212_dev $IMAGE bash
  # Disable triton_tf in /opt/tensorflow/tensorflow-source/tensorflow/BUILD
  # export HTTP_PROXY and HTTPS_PROXY in /opt/tensorflow/nvbuild.sh
  # docker exec -it tf${ver}_2212_dev bash /opt/tensorflow/nvbuild.sh --sm 6.0,6.1,7.0,7.5,8.0,8.6,8.9,9.0
  # compiling will take about 20min ~ 30min, Waiting patiently
}


prepare_tf_be() {
    ver="$1"
    TRITON_TENSORFLOW_VERSION=$ver
    TRITON_TENSORFLOW_CC_LIBNAME="libtensorflow_cc.so"
    TRITON_TENSORFLOW_FW_LIBNAME="libtensorflow_framework.so"
    # TRITON_TENSORFLOW_DOCKER_IMAGE="nvcr.io/nvidia/tensorflow:22.12-tf2-py3"

    if [[ $ver == "1" ]]; then
      CONT_NAME=tf1_2212_dev
      TRITON_TENSORFLOW_PYTHON_PATH="/usr/local/lib/python3.8/dist-packages/tensorflow_core"
      pkg_name="tensorflow-1.15.5+nv-cp38-cp38-linux_x86_64.whl"
    else 
      CONT_NAME=tf2_2212_dev
      TRITON_TENSORFLOW_PYTHON_PATH="/usr/local/lib/python3.8/dist-packages/tensorflow"
      pkg_name="tensorflow-2.10.1+nv-cp38-cp38-linux_x86_64.whl"
    fi

    OUT_DIR="./tritonbuild/third_party/libtf${ver}_v22.12"
    rm -fr ${OUT_DIR}
    LIB_DIR="${OUT_DIR}/lib"
    BIN_DIR="${OUT_DIR}/bin"
    INC_DIR="${OUT_DIR}/include"

    mkdir -p ${LIB_DIR} || echo "ignore..." || true
    mkdir -p ${BIN_DIR} || echo "ignore..." || true
    mkdir -p ${INC_DIR} || echo "ignore..." || true
    mkdir -p ${LIB_DIR}/tf_backend_deps || echo "ignore..." || true
    mkdir -p ${OUT_DIR}/pkg || echo "ignore..." || true

    LIBS_ARCH=x86_64
    TRITON_TENSORFLOW_INSTALL_EXTRA_DEPS=ON

    # docker create --name ${CONT_NAME} ${TRITON_TENSORFLOW_DOCKER_IMAGE}
    docker cp -L ${CONT_NAME}:/usr/local/lib/tensorflow/${TRITON_TENSORFLOW_CC_LIBNAME}.${TRITON_TENSORFLOW_VERSION} \
      ${LIB_DIR}/${TRITON_TENSORFLOW_CC_LIBNAME}.${TRITON_TENSORFLOW_VERSION}
    docker cp ${CONT_NAME}:${TRITON_TENSORFLOW_PYTHON_PATH}/${TRITON_TENSORFLOW_FW_LIBNAME}.${TRITON_TENSORFLOW_VERSION} \
      ${LIB_DIR}/${TRITON_TENSORFLOW_FW_LIBNAME}.${TRITON_TENSORFLOW_VERSION}

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

    if [ "${TRITON_TENSORFLOW_INSTALL_EXTRA_DEPS}" = "ON" ]; then 
      docker exec ${CONT_NAME} sh -c  "tar -cf - /usr/lib/${LIBS_ARCH}-linux-gnu/libnccl.so*" | tar --strip-components=3 -xf - -C ${LIB_DIR}/tf_backend_deps 
    fi

    docker cp ${CONT_NAME}:/tmp/pip/${pkg_name} ${OUT_DIR}/pkg/

    pushd ${LIB_DIR}
    ln -s libtensorflow_cc.so.${TRITON_TENSORFLOW_VERSION} libtensorflow_cc.so
    ln -s libtensorflow_framework.so.${TRITON_TENSORFLOW_VERSION} libtensorflow_framework.so
    popd

    echo "Extracting ${TRITON_TENSORFLOW_CC_LIBNAME}.${TRITON_TENSORFLOW_VERSION} and ${TRITON_TENSORFLOW_FW_LIBNAME}.${TRITON_TENSORFLOW_VERSION} from ${CONT_NAME}"
}


## libtorch
# create_pt_compile_env
# prepare_pt_be

## libtf
# create_tf_compile_env 1
# create_tf_compile_env 2
# prepare_tf_be 2
# prepare_tf_be 1

## libonnxruntime
