#!/bin/bash

function install() {
    fastllm_path=/home/hanfeng/projects/fastllm
    echo ${fastllm_path}
    cd $fastllm_path
    mkdir build
    cd build
    cmake .. -DUSE_CUDA=ON # 如果不使用GPU编译，那么使用 cmake .. -DUSE_CUDA=OFF
    make -j
    cd tools && python3 setup.py install
    # patch
    dist_pkg_dir="/usr/local/lib/python3.8/dist-packages"
    mv /usr/lib/python3.8/site-packages/fastllm_pytools-0.0.1-py3.8.egg $dist_pkg_dir
    mv /usr/lib/python3.8/site-packages/easy-install.pth $dist_pkg_dir
}

install
