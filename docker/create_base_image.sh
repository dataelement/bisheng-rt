#!/bin/bash


function prepare() {
  # prepare zh env
  apt update && apt-get -y install language-pack-zh-hans
  localedef -c -f UTF-8 -i zh_CN zh_CN.utf8
  export LC_ALL=zh_CN.utf8

  # install system libs
  apt install -y nasm
  apt install -y zlib1g-dev
  apt install -y rapidjson-dev
  apt install -y libssl-dev
  apt install -y libboost1.71-dev
  apt install -y libre2-dev
  apt install -y libb64-dev
  apt install -y libarchive-dev

  # install cv libs
  apt install -y libsm6 libxext6 libxrender-dev libgl1

  # install python env
  apt install -y python3.8 libpython3.8-dev python3-pip
  local repo="https://mirrors.aliyun.com/pypi/simple"
  pip3 install --upgrade wheel setuptools -i repo
  pip3 install --upgrade numpy -i $repo

  # install dcgm
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
  wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb
  dpkg -i cuda-keyring_1.0-1_all.deb
  apt-get update
  apt-get install -y datacenter-gpu-manager
  rm cuda-keyring_1.0-1_all.deb

  # install cmake
  wget http://192.168.106.8:8100/deps/cmake-3.23.1-linux-x86_64.tar.gz
  tar zxf cmake-3.23.1-linux-x86_64.tar.gz --strip-components 1 -C /usr/local
  rm cmake-3.23.1-linux-x86_64.tar.gz
}


function run_cuda118_dev() {
  MOUNT="-v $HOME:$HOME -v /home/public:/home/public"
  IMAGE="nvcr.io/nvidia/tritonserver:22.12-py3-min"
  CNT="bisheng_rt_dev_base"
  docker run --gpus=all --net=host -itd --shm-size=10G \
    --name ${CNT} ${MOUNT} $IMAGE bash
}


function commit_dev_image() {
    build_image="dataelement/bisheng-rt-base:0.0.1"
    CNT="bisheng_rt_dev_base"
    docker rmi ${build_image}
    docker commit -a "author@dataelem.com" -m "commit bisheng-rt base dev image" \
      ${CNT} ${build_image}
}


# run_cuda118_dev
# prepare
# commit_dev_image