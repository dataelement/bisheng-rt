#!/bin/bash


function start_build_image() {
    docker run -itd --rm --name llamacpp_build -v $HOME:$HOME ubuntu:20.04 bash
}

function compile() {
    # From ubuntu:20.04
    # apt update && apt install -y build-essential python3 python3-pip git wget libopenblas-dev libopenblas-base pkg-config
    # wget http://192.168.106.8:8100/deps/cmake-3.26.3-linux-x86_64.tar.gz
    # tar zxf cmake-3.26.3-linux-x86_64.tar.gz --strip-components 1 -C /usr/local
    # rm cmake-3.26.3-linux-x86_64.tar.gz

    # apt-get install -y --no-install-recommends libopenblas-dev

    llamacpp_path=/public/bisheng/release/dist/llama-cpp-python/vendor/llama.cpp
    echo ${llamacpp_path}
    cd $llamacpp_path
    mkdir build
    cd build
    cmake .. -DLLAMA_BLAS=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS \
        -DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libopenblas.so.0 \
        -DLLAMA_BUILD_TESTS=OFF

    # cmake .. -DLLAMA_BUILD_TESTS=OFF
    cmake --build . --config Release
}


function compile2() {
  apt install -y libopenblas-dev libopenblas-base pkg-config
  pushd /public/bisheng/release/dist/llama-cpp-python
  CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install .

  # pip install .
}


function gen_libs() {
    echo "gen libs"
}

function llama2_quan4() {
  LLAMACPP_LIB="/public/bisheng/release/third_party/libllamacpp_v0.1"
  REQTXT="${LLAMACPP_LIB}/bin/requirements.txt"
  # pip3 install -r ${REQTXT}
  # pip3 install starlette==0.32.0.post1 anyio==4.1.0

  QUAN=${LLAMACPP_LIB}/bin/quantize
  C1PY="${LLAMACPP_LIB}/bin/convert-hf-to-gguf.py"
  C0PY="python3 ${LLAMACPP_LIB}/bin/convert.py"

  input_dir="/public/bisheng/model_repository/Llama-2-7b-chat-hf"
  output_dir="/public/bisheng/model_repository/llama2-7b-chat-hf-4b-gguf"
  output_file="${output_dir}/ggml-model-f16.gguf"
  # $C0PY ${input_dir} --outfile $output_file
  $QUAN ${output_file} Q4_0
}


function baichuan2_quan4() {
  LLAMACPP_LIB="/public/bisheng/release/third_party/libllamacpp_v0.1"
  REQTXT="${LLAMACPP_LIB}/bin/requirements.txt"
  # pip3 install -r ${REQTXT}

  QUAN=${LLAMACPP_LIB}/bin/quantize
  C1PY="${LLAMACPP_LIB}/bin/convert-hf-to-gguf.py"
  C0PY="python3 ${LLAMACPP_LIB}/bin/convert.py"

  input_dir="/public/bisheng/model_repository/Baichuan2-13B-Chat"
  output_dir="/public/bisheng/model_repository/Baichuan2-13B-Chat-4b-gguf"
  output_file="${output_dir}/ggml-model-f16.gguf"
  $C0PY ${input_dir} --outfile $output_file
  # $QUAN ${output_file} Q4_0
}

function qwen_quan4() {
  LLAMACPP_LIB="/public/bisheng/release/third_party/libllamacpp_v0.1"
  export LD_LIBRARY_PATH=${LLAMACPP_LIB}/lib:${LD_LIBRARY_PATH}

  REQTXT="${LLAMACPP_LIB}/bin/requirements.txt"
  # pip3 install -r ${REQTXT}

  QUAN=${LLAMACPP_LIB}/bin/quantize
  C1PY="${LLAMACPP_LIB}/bin/convert-hf-to-gguf.py"
  C0PY="python3 ${LLAMACPP_LIB}/bin/convert.py"

  input_dir="/public/bisheng/model_repository/Qwen-1_8B-Chat"
  output_dir="/public/bisheng/model_repository/Qwen-1_8B-Chat-4b-gguf"
  output_file="${output_dir}/ggml-model-f16.gguf"
  # $C1PY ${input_dir} --outfile $
  $QUAN ${output_file} Q4_0
}

function bench() {
  LLAMACPP_LIB="/public/bisheng/release/third_party/libllamacpp_v0.1"
  export LD_LIBRARY_PATH=${LLAMACPP_LIB}/lib:${LD_LIBRARY_PATH}
  
  BEN="${LLAMACPP_LIB}/bin/llama-bench"
  # model_dir="/public/bisheng/model_repository/llama2-7b-chat-hf-4b-gguf"
  model_dir="/public/bisheng/model_repository/Qwen-1_8B-Chat-4b-gguf"
  model_file="${model_dir}/ggml-model-Q4_0.gguf"

  $BEN -m ${model_file} -n 0 -n 16 -p 64 -t 1,2,4,8,16,32

}


# start_build_image
# compile
# compile2
# llama2_quan4
# bench

# qwen_quan4
bench

