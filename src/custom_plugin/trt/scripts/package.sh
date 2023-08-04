#!/bin/bash

set -e

project_dir=$(cd $(dirname $0)/../;pwd)

pack_and_pub() {
    if [[ ! -d build/trt_release ]]; then
        mkdir -p build/trt_release
    fi

    mkdir -p build/trt_release/bin
    mkdir -p build/trt_release/libs
    mkdir -p build/trt_release/config
    mkdir -p build/trt_release/script

    cp samples/sampleTrtMaskRcnn/mrcnn_config.json build/trt_release/config/
    cp samples/sampleTrtTransformer/transformer_config.json build/trt_release/config/
    cp samples/sampleTrtMaskRcnn/python/convert_mrcnn_weights.py build/trt_release/script/
    cp samples/sampleTrtTransformer/python/convert_trans_weights.py build/trt_release/script/

    for binary in "build/out/sample_trt_mask_rcnn" "build/out/sample_trt_transformer"; do
        if [[ -f "$binary" ]]; then
            deps=$(ldd $binary | grep "=> /" | awk -F" => " '{print $2}' | awk '{print $1}')
            for d in $deps; do
                dep_copy=true
                for sys_dep in "libpthread.so" "libm.so" "libc.so" "libdl.so" "librt.so"; do
                    if [[ $d = *"$sys_dep"* ]]; then
                        dep_copy=false
                        break
                    fi
                done
                if [[ "$dep_copy" == true ]]; then
                    cp $d build/trt_release/libs/
                fi
            done
            cp $binary build/trt_release/bin/
        else
            echo "Warning: $binary does not exist."
        fi
    done

    dt="$1"
    tar_name="${dt}.tar.gz"
    echo "tag info: ${dt}"
    echo "START TO PACK..."
    cd build
    tar czf ${tar_name} trt_release/
    cd -
    echo "SUCC TO PACK"

    scp build/${tar_name} public@m7-model-gpu08:/home/public/ftp/release/trt-lib
    echo "ftp://m7-model-gpu08:/release/trt-lib/${tar_name}"
}

case $1 in
    "pack_and_pub")
        pack_and_pub ${@:(-1):1}
        ;;
    *)
        cat << EOF
        usage: package.sh <command>

        Supported Commands:

            pack_and_pub <version>(20211209_v1.0.2_rc0-11.1)  pack nnpredictor and push to ftp
EOF
        ;;
esac
