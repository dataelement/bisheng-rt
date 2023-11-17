#!/bin/bash


function build_image() {
    curr=$(pwd)
    pushd ${curr}/docker/prod
    docker build -t dataelement/bisheng-rt:0.0.5 -f Dockerfile . --no-cache
    popd
}


build_image