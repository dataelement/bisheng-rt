#!/bin/bash

PIP_REPO="https://mirrors.tencent.com/pypi/simple"

# # 安装bisheng-rt系统库依赖
# export DEBIAN_FRONTEND=noninteractive
# apt update && apt install -y nasm zlib1g-dev libssl-dev libre2-dev libb64-dev locales

# # Configure language
# locale-gen en_US.UTF-8
# export LC_ALL=en_US.UTF-8
# export LANG=en_US.UTF-8 
# export LANGUAGE=en_US.UTF-8

# # Configure timezone
# export TZ=Asia/Shanghai
# ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# # 安装bisheng-rt
# mkdir -p /opt/bisheng-rt/
# cp -fr ./install/* /opt/bisheng-rt/
# cp -fr ./deps /opt/bisheng-rt/deps
# pushd /opt/bisheng-rt

# 安装bisheng-rt依赖
# FIX
ln -s /usr/local/bin/pip3 /usr/bin/pip3.8
pip install -r deps/requirements.txt -i $PIP_REPO

cd deps/flash-attention && \
  pip install . -i $PIP_REPO && \
  pip install csrc/layer_norm -i $PIP_REPO && \
  pip install csrc/rotary -i $PIP_REPO
