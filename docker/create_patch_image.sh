#!/bin/bash


function create_patch_container() {
    IMAGE="dataelement/bisheng-rt:0.0.5.1"
    MOUNT="-v $HOME:$HOME -v /public:/public"
    docker run --gpus=all --net=host -itd --shm-size=10G \
        --name bisheng_rt_patch_0051 ${MOUNT} $IMAGE bash
}

function patch() {
    MODEL_DIR="/usr/local/lib/python3.8/dist-packages/pybackend_libs/dataelem/model"
    docker cp /public/bisheng/patches/qwen.py bisheng_rt_patch_0051:${MODEL_DIR}/llm/
    docker cp /public/bisheng/patches/qwen_utils.py bisheng_rt_patch_0051:${MODEL_DIR}/llm/
    docker cp /public/bisheng/patches/llm.py bisheng_rt_patch_0051:${MODEL_DIR}/llm/
}

function create_patch_image() {
    docker rmi dataelement/bisheng-rt:0.0.5.2 || echo "true"
    docker commit -a "author@dataelem.com" -m "commit bisheng-rt image" \
        bisheng_rt_patch_0051 dataelement/bisheng-rt:0.0.5.2
}


# create_patch_container
# patch
create_patch_image
