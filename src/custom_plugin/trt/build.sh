build_tensorrt7() {
    export TRT_RELEASE=/root/gulixin/TensorRT-7.2.1.6
    mkdir -p build && cd build
    cmake .. -DTRT_RELEASE_DIR=$TRT_RELEASE -DTRT_BIN_DIR=`pwd`/out -DCUDA_VERSION=11.1 -DCUDNN_VERSION=8.0
    make

}

build_tensorrt8() {
    mkdir -p build && cd build
    cmake .. -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu -DTRT_INC_DIR=/usr/include/x86_64-linux-gnu -DTRT_BIN_DIR=`pwd`/out -DCUDA_VERSION=11.7 -DCUDNN_VERSION=8.5
    make

}

# build_tensorrt7
build_tensorrt8
