ARG CUDA=11.1
ARG CUDNN=cudnn8
ARG UBUNTU_VERSION=16.04

FROM nvidia/cuda:${CUDA}-${CUDNN}-devel-ubuntu${UBUNTU_VERSION}

MAINTAINER "gulixin, 4paradigm.com"

WORKDIR /

### BASIC
RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-get update
RUN apt-get install -y --no-install-recommends curl tmux vim software-properties-common
RUN apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    zlib1g-dev \
    git \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    sudo \
    ssh \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    build-essential \
    libarchive-dev \
    libsm6 \
    libxrender1 \
    libxext-dev \
    gdb \
    libjsoncpp-dev

RUN cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip

### 设置语言
RUN apt-get install locales
RUN locale-gen en_US.UTF-8
ENV LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

### 设置时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

### GPU ENV
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV C_INCLUDE_PATH=/usr/local/cuda/include:$C_INCLUDE_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

### Install Cmake
RUN cd /tmp && \
    wget ftp://m7-model-gpu08/packages/cmake-3.14.4-Linux-x86_64.sh && \
    chmod +x cmake-3.14.4-Linux-x86_64.sh && \
    ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.14.4-Linux-x86_64.sh

