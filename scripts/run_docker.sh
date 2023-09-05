#!/bin/bash

function run_dev() {
    MOUNT="-v /home/work:/home/work -v /home/public:/home/public"
    IMAGE="tritonserver:22.08"
    docker run --gpus=all --net=host -itd --shm-size=10G --name bisheng_rt_v0.0.1 ${MOUNT} $IMAGE bash
}


function build_image() {
    curr=$(pwd)
    cd ./output/install/python && find ./ -name __pycache__ -exec rm -fr {} \;
    cd $cur

    pushd $cur/output
    docker build -t dataelem/bisheng-rt:0.0.1 -f "$curr/docker/runtime.Dockerfile" .
    popd
}


run_dev
