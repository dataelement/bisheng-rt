#!/bin/bash

function merge() {
    RT_ENTERPRISE_DIR="/home/hanfeng/projects/bisheng-rt-enterprise"
    RT_ENTER_RELEASE_DIR="$RT_ENTERPRISE_DIR/tritonbuild/install"

    RT_RELEASE_DIR="/opt/bisheng-rt"
    cp $RT_ENTER_RELEASE_DIR/bin/rtserver $RT_RELEASE_DIR/bin/rtserver.ent
    cp -fr $RT_ENTER_RELEASE_DIR/lib $RT_RELEASE_DIR/bin/
    # cp -fr $RT_ENTER_RELEASE_DIR/backends $RT_RELEASE_DIR/backends_enterprise 
    # cp -fr $RT_ENTER_RELEASE_DIR/plugins $RT_RELEASE_DIR/
}

merge