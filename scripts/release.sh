#!/bin/bash

function update() {
    projdir="/home/hanfeng/projects/bisheng-rt"
    build_install_dir="${projdir}/tritonbuild/install"
    releasedir="/opt/bisheng-rt"

    rm -fr ${releasedir}/resource/internal_model_repository/*
    cp -fr ${projdir}/resource/internal_model_repository/* ${releasedir}/resource/internal_model_repository/

    rm -fr ${releasedir}/backends/python/pybackend_libs
    cp -fr ${projdir}/python/pybackend_libs/src/pybackend_libs ${releasedir}/backends/python/

    rm -fr ${releasedir}/backends_enterprise/python/pybackend_libs
    cp -fr ${projdir}/python/pybackend_libs/src/pybackend_libs ${releasedir}/backends_enterprise/python/

    cp -fr ${build_install_dir}/backends/tensorflow2/libtriton_tensorflow2.so ${releasedir}/backends/tensorflow2/

    cp -fr ${projdir}/docker/entrypoint.sh ${releasedir}/bin/
    cp -fr ${build_install_dir}/lib ${releasedir}
    cp -fr ${build_install_dir}/bin ${releasedir}

    # cp -fr ${projdir}/src/tests/python/elem_ocr/app_config.pbtxt ${releasedir}/models/model_repository/elem_ocr_collection_v3/

}

update