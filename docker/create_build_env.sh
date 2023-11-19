#!/bin/bash


function create_build_env() {
    MOUNT="-v $HOME:$HOME -v /public:/public"
    IMAGE="dataelement/bisheng-rt-base:0.0.1"
    docker pull $IMAGE
    docker run --gpus=all --net=host -itd --shm-size=10G \
        --name bisheng_rt_build ${MOUNT} $IMAGE bash

    # patch dependences
    cmd="apt update && apt install -y libarchive-dev patchelf libgl1 libjsoncpp-dev pkg-config"
    docker exec -it bisheng_rt_build sh -c "$cmd"
}


create_build_env
