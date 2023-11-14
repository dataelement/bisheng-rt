#!/bin/bash


function hot_patch() {
  patch_tar="bisheng-rt-v0.0.4-patches.tar.gz"
  patch_url="https://public2:qTongs8YdIwXSRPX@nexus.dataelem.com/repository/product/bisheng/${patch_tar}"
  echo $patch_url
  wget --no-check-certificate $patch_url 
  tar zxf $patch_tar

  rt_container="$1"
  patch1="bisheng-rt-v0.0.4-patches/pybackend_libs/__init__.py"

  echo "Patch Files: $patch1 Container Name: ${rt_container}"
  docker cp ${patch1} ${rt_container}:/opt/bisheng-rt/backends/python/pybackend_libs/dataelem/model/
  docker cp ${patch1} ${rt_container}:/opt/bisheng-rt/backends_enterprise/python/pybackend_libs/dataelem/model/
  docker restart ${rt_container}
  
  rm ${patch_tar}
  rm -fr bisheng-rt-v0.0.4-patches/
  echo "Patch succeed"
}


hot_patch $1