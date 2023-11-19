#!/bin/bash


function package() {
  ver="$1"
  projdir="$(cd $(dirname $0)/..; pwd)"
  releasedir="${projdir}/output"
  if [ ! -d ${releasedir} ]; then
    mkdir -p $releasedir
  else
    rm -fr ${releasedir}
    mkdir -p ${releasedir}
  fi

  install_dir="${projdir}/tritonbuild/install"
  pub_file="bisheng-rt-${ver}.tar.gz"

  pushd ${projdir}/tritonbuild
  tar zcf ${releasedir}/${pub_file} ./install
  echo "upload to file repository"
  upload-file-prod ${releasedir}/${pub_file} pub
  popd
}


package $1
