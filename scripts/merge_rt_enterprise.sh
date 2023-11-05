#!/bin/bash

function merge() {
    RT_ENTERPRISE_DIR="/home/hanfeng/projects3/bisheng-rt-enterprise"
    RT_ENTER_RELEASE_DIR="$RT_ENTERPRISE_DIR/tritonbuild/install"

    RT_RELEASE_DIR="/opt/bisheng-rt"
    cp $RT_ENTER_RELEASE_DIR/bin/rtserver $RT_RELEASE_DIR/bin/rtserver.enter
    cp -fr $RT_ENTER_RELEASE_DIR/lib $RT_RELEASE_DIR/bin/
    cp -fr $RT_ENTER_RELEASE_DIR/backends $RT_RELEASE_DIR/backends_enterprise
}

merge