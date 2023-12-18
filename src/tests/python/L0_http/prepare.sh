#!/bin/bash

function prepare() {
    # MOUNT="-v $HOME:$HOME -v /public:/public"
    # IMAGE="dataelement/bisheng-rt-runtime:0.0.1"
    # docker run --gpus=all --net=host -itd --shm-size=10G \
    #     --name bisheng_rt_test ${MOUNT} $IMAGE bash

    cmd="ln -s $HOME/projects/bisheng-rt/tritonbuild/install /opt/bisheng-rt"
    docker exec bisheng_rt_test $cmd
    cmd2="ln -s $HOME/projects/bisheng-rt/src/tests/python /opt/bisheng-rt/tests"
    docker exec bisheng_rt_test $cmd2
}

prepare
