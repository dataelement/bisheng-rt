#!/bin/bash

function run_dev() {
    MOUNT="-v /home/hanfeng:/home/hanfeng"
    IMAGE="tritonserver:22.08"
    docker run --gpus=all --net=host -itd --shm-size=10G --name bisheng_rt_dev ${MOUNT} $IMAGE bash
}

run_dev
