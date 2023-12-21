#!/bin/bash


function create_prod_base_image() {
    IMAGE="dataelement/bisheng-rt-base:0.0.1"
    MOUNT="-v $HOME:$HOME"
    docker run --gpus=all --net=host -itd --shm-size=10G \
        --name bisheng-rt-runtime-dev ${MOUNT} $IMAGE bash
}


function create_runtime_image() {
    IMAGE="dataelement/bisheng-rt-runtime:0.0.1"
    MOUNT="-v $HOME:$HOME -v /home/public:/home/public"
    docker run --gpus=all --net=host -itd --shm-size=10G \
        --name bisheng-rt-runtime-v001 ${MOUNT} $IMAGE bash
}


function test_run() {
    PIP_REPO=https://mirrors.tencent.com/pypi/simple
    NEXUS_REPO="https://public2:qTongs8YdIwXSRPX@nexus.dataelem.com/repository/product/bisheng"

    EXTRA_PIP_REPO="http://public:26rS9HRxDqaVy5T@nx.dataelem.com/repository/pypi-hosted/simple"
    # apt update && apt install libarchive-dev patchelf libgl1 libjsoncpp-dev -y

    # # Configure language
    # locale-gen en_US.UTF-8
    # export LC_ALL=en_US.UTF-8
    # export LANG=en_US.UTF-8
    # export LANGUAGE=en_US.UTF-8


    # # Configure timezone
    # export TZ=Asia/Shanghai
    # ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
    # ln -s /usr/local/bin/pip3 /usr/bin/pip3.8

    # # pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
    # # pip install torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

    # # # wget ${NEXUS_REPO}/torch-2.0.1+cu118-cp38-cp38-linux_x86_64.whl
    # pip install ./docker/deps/torch-2.0.1+cu118-cp38-cp38-linux_x86_64.whl -i $PIP_REPO
    # pip install torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
    # pip install ./docker/deps/tensorflow-1.15.5+nv-cp38-cp38-linux_x86_64.whl -i $PIP_REPO

    # wget ${NEXUS_REPO}/flash-attention-v2.3.3.tar.gz && tar zxf flash-attention-v2.3.3.tar.gz
    
    # build is very slowly, be patiently, about 20-30mins
    # pushd ./docker/deps/flash-attention
    # pip install packaging
    # MAX_JOBS=10 pip install . -i $PIP_REPO && \
    # MAX_JOBS=10 pip install csrc/layer_norm -i $PIP_REPO
    # popd

    pip install lanms==1.0.2 -i ${EXTRA_PIP_REPO}
    # pip install -r python/pybackend_libs/requirements.txt -i $PIP_REPO
    # echo "clean" 
    apt-get clean &&  rm -rf /var/lib/apt/lists/* && rm -rf /root/.cache/pip
}


function temp_build_image() {
    docker rmi dataelement/bisheng-rt-runtime:0.0.1
    docker commit -a "author@dataelem.com" -m "commit bisheng-rt runtime image" \
        bisheng-rt-runtime-dev dataelement/bisheng-rt-runtime:0.0.1
        
    # docker save -o ./bisheng-rt-runtime-0.0.1.tar dataelement/bisheng-rt-runtime:0.0.1
}


function build_image() {
    curr=$(pwd)
    pushd ${curr}/python
    docker build -t dataelement/bisheng-rt-runtime:0.0.1 \
        -f "$curr/docker/prod-base.Dockerfile" . --no-cache
    popd
}


function update_torch() {
    PIP_REPO=https://mirrors.tencent.com/pypi/simple
    EXTRA_PIP_REPO="https://public:26rS9HRxDqaVy5T@nx.dataelem.com/repository/pypi-hosted/simple"

    # pip install torch==2.1.2 -i ${EXTRA_PIP_REPO} --extra-index-url ${PIP_REPO}

    # LOCAL_PKG="/public/bisheng/release/dist/torch-2.1.2+cu118-cp38-cp38-linux_x86_64.whl"
    # pip3 install $LOCAL_PKG -i https://mirrors.tencent.com/pypi/simple

    # pip install torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
    # pip3 install -U xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118
    # pip3 install vllm==0.2.6 -i ${EXTRA_PIP_REPO} --extra-index-url ${PIP_REPO}

    
    # wget ${NEXUS_REPO}/flash-attention-v2.3.3.tar.gz && tar zxf flash-attention-v2.3.3.tar.gz


    # build is very slowly, be patiently, about 20-30mins
    pushd /public/bisheng/release/dist/flash-attention
    pip install packaging
    MAX_JOBS=10 pip install . -i $PIP_REPO

    #MAX_JOBS=10 pip install csrc/layer_norm -i $PIP_REPO
    popd

}


# test_run
# create_prod_base_image
# temp_build_image
# create_runtime_image
update_torch