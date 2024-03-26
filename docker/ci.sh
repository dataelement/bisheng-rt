#!/bin/bash


function build_rt() {
    tag=$1
    cd $HOME/cicd/bisheng-rt && git checkout main && git pull && git checkout $tag && cd -
    if [[ $? -ne 0 ]]; then
        echo "checkout bisheng-rt $tag failed"
        exit 1
    fi

    if [[ ! -d $HOME/build-logs ]]; then
        mkdir -p $HOME/build-logs
    fi

    cmd="bash $HOME/cicd/bisheng-rt/docker/build_helper.sh server"
    docker exec bisheng_rt_build $cmd > $HOME/build-logs/build-rt-server.log 2>&1
    if [[ $? -ne 0 ]]; then
        echo "failed to build server"
        exit 1
    else
        echo '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
        echo "succ to build server"
    fi

    cmd2="bash $HOME/cicd/bisheng-rt/docker/build_helper.sh backends"
    docker exec bisheng_rt_build $cmd2 > $HOME/build-logs/build-rt-backends.log 2>&1
    if [[ $? -ne 0 ]]; then
        echo "failed to build backends"
        exit 1
    else
        echo '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
        echo "succ to build backend"
    fi

    bash $HOME/cicd/bisheng-rt/docker/release.sh ${tag:1}
    if [[ $? -eq 0 ]]; then
      echo '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
      echo "succ to build bisheng-rt"
    else
      echo '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'
      echo "failed to build bisheng-rt"
    fi
}


build_rt $1
