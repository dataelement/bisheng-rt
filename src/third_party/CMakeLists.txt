# Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.17)
project(tritonthirdparty LANGUAGES C CXX)

#
# ExternalProject Dependencies
#
# For each ExternalProject dependency we use PATCH_COMMAND to copy the
# project source into TRITON_THIRD_PARTY_SRC_INSTALL_PREFIX. If
# TRITON_THIRD_PARTY_SRC_INSTALL_PREFIX is not set then source will
# not be installed. Note that PATCH_COMMAND is executed after the
# source is downloaded/updated and is not defined by default so it
# allows a good point to perform the source install.
#
include(ExternalProject)

if(NOT TRITON_THIRD_PARTY_INSTALL_PREFIX)
  message(FATAL_ERROR "TRITON_THIRD_PARTY_INSTALL_PREFIX must be set")
endif() # TRITON_THIRD_PARTY_INSTALL_PREFIX

if(NOT TRITON_THIRD_PARTY_SRC_INSTALL_PREFIX)
  message(STATUS "TRITON_THIRD_PARTY_SRC_INSTALL_PREFIX not set")
  set(INSTALL_SRC_DEST_ARG "")
else()
  set(INSTALL_SRC_DEST_ARG --dest ${TRITON_THIRD_PARTY_SRC_INSTALL_PREFIX})
endif() # TRITON_THIRD_PARTY_INSTALL_PREFIX

# If CMAKE_TOOLCHAIN_FILE is set, propagate that hint path to the external
# projects.
set(_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE "")
if (CMAKE_TOOLCHAIN_FILE)
  set(_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE "-DCMAKE_TOOLCHAIN_FILE:PATH=${CMAKE_TOOLCHAIN_FILE}")
endif()

# If VCPKG_TARGET_TRIPLET is set, propagate that hint path to the external
# projects.
set(_CMAKE_ARGS_VCPKG_TARGET_TRIPLET "")
if (VCPKG_TARGET_TRIPLET)
  set(_CMAKE_ARGS_VCPKG_TARGET_TRIPLET "-DVCPKG_TARGET_TRIPLET:STRING=${VCPKG_TARGET_TRIPLET}")
endif()

# If OPENSSL_ROOT_DIR is set, propagate that hint path to the external
# projects with OpenSSL dependency.
set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "")
if (OPENSSL_ROOT_DIR)
  set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "-DOPENSSL_ROOT_DIR:PATH=${OPENSSL_ROOT_DIR}")
endif()

set(HTTP_SERVER "192.168.106.8:8100")
set(THIRD_PARTY_PREFIX "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/common")

#
# Build libcurl
#
ExternalProject_Add(curl
  PREFIX curl
  GIT_REPOSITORY "https://github.com/curl/curl.git"
  GIT_TAG "curl-7_66_0"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/curl/src/curl"
  EXCLUDE_FROM_ALL ON
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DBUILD_TESTING:BOOL=OFF
    -DBUILD_CURL_EXE:BOOL=OFF
    -DBUILD_SHARED_LIBS:BOOL=OFF
    -DCURL_STATICLIB:BOOL=ON
    -DHTTP_ONLY:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/curl
  PATCH_COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/install_src.py --src <SOURCE_DIR> ${INSTALL_SRC_DEST_ARG}
)

# curlpp
ExternalProject_Add(
    curlpp
    PREFIX curlpp
    URL http://192.168.106.8/deps/curlpp-0.8.1.zip
    URL_HASH SHA256=67bb923bee565d1076baa6a758d299594ff0d8fd26fc5e02b83c5f5b5764ccee
    BUILD_IN_SOURCE 0
    INSTALL_DIR ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/curlpp
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_LIBDIR=lib
        -DCURL_INCLUDE_DIR=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/curl/include
        -DCURL_LIBRARY=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/curl/lib
        -DCMAKE_PREFIX_PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/curlpp
        -DCMAKE_INSTALL_PREFIX=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/curlpp
        -DCMAKE_CXX_FLAGS=-fPIC
        -DCMAKE_C_FLAGS=-fPIC   
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --parallel 10
    DEPENDS curl
)

#
# Get the protobuf and grpc source used for the GRPC endpoint. We must
# use v1.25.0 because later GRPC has significant performance
# regressions (e.g. resnet50 bs128).
#
ExternalProject_Add(grpc-repo
  PREFIX grpc-repo
  # GIT_REPOSITORY "https://github.com/grpc/grpc.git"
  # GIT_TAG "v1.48.0"
  URL http://${HTTP_SERVER}/pkgs/grpc-v1.48.0.tar.gz
  URL_HASH SHA256=3b42a23d00ad3eff4ed0fc85d54435edd6e2222033a4140262975c19eacd39c3
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/grpc-repo/src/grpc"
  EXCLUDE_FROM_ALL ON
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
  PATCH_COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/install_src.py --src <SOURCE_DIR> ${INSTALL_SRC_DEST_ARG} --dest-basename=grpc_1.48.0
)

#
# Build nlohmann/json
#
ExternalProject_Add(nlohmann-json
  PREFIX nlohmann-json
  GIT_REPOSITORY "https://github.com/nlohmann/json.git"
  GIT_TAG "v3.9.0"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/json"
  EXCLUDE_FROM_ALL ON
  CMAKE_CACHE_ARGS
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=TRUE
    -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/nlohmann_json
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DBUILD_SHARED_LIBS:STRING=yes
    -DBUILD_TESTING:BOOL=OFF
    # -H. -Bcmake-out
  PATCH_COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/install_src.py --src <SOURCE_DIR> ${INSTALL_SRC_DEST_ARG} --dest-basename=nlohmann-json
)

#
# Build protobuf project from grpc-repo, but apply a patch that is available
# in 3.20.0, to resolve a linking issue on Windows
#
# Source is from grpc-repo so don't need PATCH_COMMAND to install
# source for this target.
#
# Use -DCMAKE_BUILD_TYPE:STRING=RELEASE to workaround the protobuf build issue
# described in https://stackoverflow.com/a/60218066/2091555
#
# Use -DCMAKE_INSTALL_LIBDIR:STRING=lib to workaround a build issue on centos/rhel where it defaults to lib64
# described in https://stackoverflow.com/q/68974636
#
ExternalProject_Add(protobuf
  PREFIX protobuf
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/grpc-repo/src/grpc/third_party/protobuf/cmake"
  EXCLUDE_FROM_ALL ON
  DOWNLOAD_COMMAND ""
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -Dprotobuf_BUILD_TESTS:BOOL=OFF
    -Dprotobuf_WITH_ZLIB:BOOL=OFF
    -Dprotobuf_MSVC_STATIC_RUNTIME:BOOL=OFF
    -DCMAKE_BUILD_TYPE:STRING=RELEASE
    -DBUILD_SHARED_LIBS:STRING=no
    -DCMAKE_INSTALL_LIBDIR:STRING=lib
    -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/protobuf
  PATCH_COMMAND git cherry-pick -n f180289c4670ca1afde5865bb8a7f2b61a3efcc5
  DEPENDS grpc-repo
)

#
# Build absl (abseil) project from grpc-repo
#
# Source is from grpc-repo so don't need PATCH_COMMAND to install
# source for this target.
#
ExternalProject_Add(absl
  PREFIX absl
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/grpc-repo/src/grpc/third_party/abseil-cpp"
  EXCLUDE_FROM_ALL ON
  DOWNLOAD_COMMAND ""
  CMAKE_CACHE_ARGS
    -DCMAKE_CXX_STANDARD=14
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=TRUE
    -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/absl
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DBUILD_TESTING:BOOL=OFF
  DEPENDS grpc-repo
)

#
# Build googletest project from grpc-repo
#
# Source is from grpc-repo so don't need PATCH_COMMAND to install
# source for this target.
#
ExternalProject_Add(googletest
  PREFIX protobuf
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/grpc-repo/src/grpc/third_party/googletest"
  EXCLUDE_FROM_ALL ON
  DOWNLOAD_COMMAND ""
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/googletest
  DEPENDS grpc-repo
)

# Location where protobuf-config.cmake will be installed varies by
# platform
if (WIN32)
  set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/protobuf/cmake")
else()
  set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/protobuf/lib/cmake/protobuf")
endif()

#
# Build c-ares project from grpc-repo
#
# Source is from grpc-repo so don't need PATCH_COMMAND to install
# source for this target.
#
ExternalProject_Add(c-ares
  PREFIX c-ares
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/grpc-repo/src/grpc/third_party/cares/cares"
  EXCLUDE_FROM_ALL ON
  DOWNLOAD_COMMAND ""
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DCARES_SHARED:BOOL=OFF
    -DCARES_STATIC:BOOL=ON
    -DCARES_STATIC_PIC:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/c-ares
  DEPENDS grpc-repo
)

#
# Build GRPC
#
# Source is from grpc-repo so don't need PATCH_COMMAND to install
# source for this target.
#
ExternalProject_Add(grpc
  PREFIX grpc
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/grpc-repo/src/grpc"
  EXCLUDE_FROM_ALL ON
  DOWNLOAD_COMMAND ""
  CMAKE_CACHE_ARGS
    -DgRPC_INSTALL:BOOL=ON
    -DgRPC_BUILD_TESTS:BOOL=OFF
    -DgRPC_PROTOBUF_PROVIDER:STRING=package
    -DgRPC_PROTOBUF_PACKAGE_TYPE:STRING=CONFIG
    -DgRPC_ZLIB_PROVIDER:STRING=package
    -DgRPC_CARES_PROVIDER:STRING=package
    -DgRPC_SSL_PROVIDER:STRING=package
    -Dc-ares_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/c-ares/lib/cmake/c-ares
    -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
    ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/grpc
  DEPENDS grpc-repo c-ares protobuf absl
)

#
# Build libevent
#
ExternalProject_Add(libevent
  PREFIX libevent
  GIT_REPOSITORY "https://github.com/libevent/libevent.git"
  GIT_TAG "release-2.1.8-stable"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/libevent/src/libevent"
  EXCLUDE_FROM_ALL ON
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/libevent
  PATCH_COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/install_src.py --src <SOURCE_DIR> ${INSTALL_SRC_DEST_ARG}
)

#
# Build patched libevhtp
#
if (WIN32)
# FIXME evhtp can't be built successfully in native Windows environment
# CMAKE_PREFIX_PATH is needed to find libevent
ExternalProject_Add(libevhtp
  PREFIX libevhtp
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libevhtp/libevhtp"
  EXCLUDE_FROM_ALL ON
  DOWNLOAD_COMMAND ""
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DEVHTP_DISABLE_REGEX:BOOL=ON
    -DEVHTP_DISABLE_SSL:BOOL=ON
    -DEVHTP_TRITON_ENABLE_HTTP_CONTIGUOUS:BOOL=ON
    -DEVHTP_TRITON_ENABLE_TRACING:BOOL=${TRITON_ENABLE_TRACING}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_PREFIX_PATH:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/libevent
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/libevhtp
  PATCH_COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/install_src.py --src <SOURCE_DIR> ${INSTALL_SRC_DEST_ARG}
  DEPENDS libevent
)
else()
ExternalProject_Add(libevhtp
  PREFIX libevhtp
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libevhtp/libevhtp"
  EXCLUDE_FROM_ALL ON
  DOWNLOAD_COMMAND ""
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DEVHTP_DISABLE_REGEX:BOOL=ON
    -DEVHTP_DISABLE_SSL:BOOL=ON
    -DEVHTP_TRITON_ENABLE_HTTP_CONTIGUOUS:BOOL=ON
    -DEVHTP_TRITON_ENABLE_TRACING:BOOL=${TRITON_ENABLE_TRACING}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DLibevent_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/libevent/lib/cmake/libevent
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/libevhtp
  PATCH_COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/install_src.py --src <SOURCE_DIR> ${INSTALL_SRC_DEST_ARG}
  DEPENDS libevent
)
endif()

#
# Build Prometheus C++ library
#
# Use -DCMAKE_INSTALL_LIBDIR:STRING=lib to workaround a build issue on centos/rhel where it defaults to lib64
# described in https://stackoverflow.com/q/68974636
#
ExternalProject_Add(prometheus-cpp
  PREFIX prometheus-cpp
  #URL "https://github.com/jupp0r/prometheus-cpp/archive/v0.7.0.tar.gz"
  URL "http://192.168.106.8/pkgs/prometheus-cpp-1.0.1.tar.gz"
  URL_HASH SHA256=593e028d401d3298eada804d252bc38d8cab3ea1c9e88bcd72095281f85e6d16
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/prometheus-cpp/src/prometheus-cpp"
  EXCLUDE_FROM_ALL ON
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DENABLE_PUSH:BOOL=OFF
    -DENABLE_PULL:BOOL=OFF
    -DENABLE_TESTING:BOOL=OFF
    -DUSE_THIRDPARTY_LIBRARIES:BOOL=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_LIBDIR:STRING=lib
    -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/prometheus-cpp
  PATCH_COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/install_src.py --src <SOURCE_DIR> ${INSTALL_SRC_DEST_ARG}
)

#
# Build crc32c
#
# Use a specific hash to get ARM64 fix not included in latest release
# (https://github.com/google/crc32c/pull/52).
#
ExternalProject_Add(crc32c
  PREFIX crc32c
  GIT_REPOSITORY "https://github.com/google/crc32c.git"
  GIT_TAG "b9d6e825a1e6783195a6051639179152dac70b3b"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/crc32c/src/crc32c"
  EXCLUDE_FROM_ALL ON
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DCRC32C_BUILD_TESTS:BOOL=OFF
    -DBUILD_SHARED_LIBS:STRING=no
    -DCRC32C_BUILD_BENCHMARKS:BOOL=OFF
    -DCRC32C_USE_GLOG:BOOL=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/crc32c
  PATCH_COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/install_src.py --src <SOURCE_DIR> ${INSTALL_SRC_DEST_ARG}
)

# This is necessary because google-cloud-cpp
# does not pass protobuf_DIR and gRPC_DIR etc to its
# external projects but expects them in CMAKE_PREFIX_PATH.
#
# absl must preceed grpc to ensure that the absl from the newer grpc
# repo is used and not the absl from the older grpc.
set(GCS_CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}
   ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/c-ares/lib/cmake/c-ares
   ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/absl/lib/cmake/absl
   ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/crc32c/lib/cmake/Crc32c
   ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/grpc/lib/cmake/grpc
   ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/nlohmann_json/lib/cmake/nlohmann_json
   ${_FINDPACKAGE_PROTOBUF_CONFIG_DIR})

#
# Build google-cloud-cpp
#
ExternalProject_Add(google-cloud-cpp
  PREFIX google-cloud-cpp
  GIT_REPOSITORY "https://github.com/googleapis/google-cloud-cpp.git"
  GIT_TAG "v1.42.0"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/google-cloud-cpp/src/google-cloud-cpp"
  EXCLUDE_FROM_ALL ON
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DGOOGLE_CLOUD_CPP_DEPENDENCY_PROVIDER:STRING=package
    -DBUILD_TESTING:BOOL=OFF
    -DCMAKE_PREFIX_PATH:PATH=${GCS_CMAKE_PREFIX_PATH}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/google-cloud-cpp
    -Dnlohmann_json_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/nlohmann_json/lib/cmake/nlohmann_json
    -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
    -DCrc32c_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/crc32c/lib/cmake/Crc32c
  PATCH_COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/install_src.py --src <SOURCE_DIR> ${INSTALL_SRC_DEST_ARG}
  DEPENDS grpc c-ares protobuf crc32c absl nlohmann-json
)

#
# Build azure-storage-cpplite
#
ExternalProject_Add(azure-storage-cpplite
  PREFIX azure-storage-cpplite
  GIT_REPOSITORY "https://github.com/Azure/azure-storage-cpplite.git"
  GIT_TAG "v0.3.0"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/azure-storage-cpplite/src/azure-storage-cpplite"
  EXCLUDE_FROM_ALL ON
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DBUILD_SHARED_LIBS:BOOL=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCURL_INCLUDE_DIR:STRING=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/curl/include
    -DCURL_LIBRARY:STRING=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/curl/lib
    -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/azure-storage-cpplite
  PATCH_COMMAND git cherry-pick --no-commit a3a9c21de3fa9932dcfceb2b6617757a511c75c3 && python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/install_src.py --src <SOURCE_DIR> ${INSTALL_SRC_DEST_ARG}
  DEPENDS curl
)

#
# Build CNMeM (CUDA memory management library)
#
ExternalProject_Add(cnmem
  PREFIX cnmem
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cnmem/cnmem"
  EXCLUDE_FROM_ALL ON
  DOWNLOAD_COMMAND ""
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/cnmem
  PATCH_COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/install_src.py --src <SOURCE_DIR> ${INSTALL_SRC_DEST_ARG}
)

#
# Build aws-sdk-cpp
#
ExternalProject_Add(aws-sdk-cpp
  PREFIX aws-sdk-cpp
  GIT_REPOSITORY "https://github.com/aws/aws-sdk-cpp.git"
  GIT_TAG "1.8.186"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/aws-sdk-cpp/src/aws-sdk-cpp"
  EXCLUDE_FROM_ALL ON
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DBUILD_ONLY:STRING=s3
    -DBUILD_SHARED_LIBS:BOOL=OFF
    -DMINIMIZE_SIZE:BOOL=ON
    -DENABLE_TESTING:BOOL=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp
  PATCH_COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/install_src.py --src <SOURCE_DIR> ${INSTALL_SRC_DEST_ARG}
)

#
# Build opencv-3.4.2
#

ExternalProject_Add(
    zlib
    PREFIX zlib
    URL http://${HTTP_SERVER}/deps/zlib-1.2.8.tar.gz
    URL_HASH SHA256=36658cb768a54c1d4dec43c3116c27ed893e88b02ecfcb44f2166f9c0b7f2a0d
    CONFIGURE_COMMAND CFLAGS=-fPIC CXXFLAGS=-fPIC LDFLAGS=-fPIC ./configure --prefix=${THIRD_PARTY_PREFIX} 
    BUILD_COMMAND CFLAGS=-fPIC CXXFLAGS=-fPIC LDFLAGS=-fPIC make -j $ENV{J}
    INSTALL_COMMAND make install
    BUILD_IN_SOURCE 1
)
# libpng
ExternalProject_Add(
    libpng
    PREFIX libpng
    URL http://${HTTP_SERVER}/deps/libpng-1.6.37.tar.gz
    URL_HASH SHA256=daeb2620d829575513e35fecc83f0d3791a620b9b93d800b763542ece9390fb4
    BUILD_IN_SOURCE 0
    INSTALL_DIR ${THIRD_PARTY_PREFIX}
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_LIBDIR=lib
        -DCMAKE_PREFIX_PATH=${THIRD_PARTY_PREFIX}
        -DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_PREFIX}
        -DPNG_TESTS:BOOL=OFF
        -DPNG_SHARED:BOOL=OFF
        -DPNG_STATIC:BOOL=ON
        -DPNG_DEBUG:BOOL=OFF
        -DCMAKE_CXX_FLAGS=-fPIC
        -DCMAKE_C_FLAGS=-fPIC
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --parallel $ENV{J}
    INSTALL_COMMAND make install &&
        cd ${THIRD_PARTY_PREFIX}/include && ln -s libpng16 libpng
    DEPENDS zlib
)

# libjpeg-turbo-202
ExternalProject_Add(
    libjpeg-turbo-202
    PREFIX libjpeg-turbo-202
    URL http://${HTTP_SERVER}/deps/libjpeg-turbo-2.0.2.tar.gz
    URL_HASH SHA256=b45255bd476c19c7c6b198c07c0487e8b8536373b82f2b38346b32b4fa7bb942
    BUILD_IN_SOURCE 0
    CMAKE_ARGS
        -G "Unix Makefiles"
        -DCMAKE_CXX_FLAGS=-fPIC
        -DCMAKE_C_FLAGS=-fPIC
        -DENABLE_SHARED=1
        -DENABLE_STATIC=1
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_LIBDIR=${THIRD_PARTY_PREFIX}/lib
        -DCMAKE_PREFIX_PATH=${THIRD_PARTY_PREFIX}
        -DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_PREFIX}
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --parallel $ENV{J}
)

# opencv-342
ExternalProject_Add(
    opencv-342
    PREFIX opencv-342
    URL http://${HTTP_SERVER}/deps/opencv-3.4.2.tar.gz
    URL_HASH SHA256=81dbd5e7e9f8a4c936b94629bf4765745942a1d634ae38ec08bc57b73b28ffc5
    BUILD_IN_SOURCE 0
    CMAKE_ARGS
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_PNG=OFF
        -DBUILD_JASPER=ON
        -DBUILD_JPEG=OFF
        -DBUILD_WEBP=ON
        -DBUILD_TIFF=ON
        -DBUILD_ZLIB=ON
        -DWITH_JPEG=ON
        -DWITH_WEBP=ON
        -DWITH_PNG=ON
        -DWITH_JASPER=ON
        -DWITH_TIFF=ON
        -DWITH_LIBV4L=ON
        -DWITH_V4L=OFF
        -DWITH_FFMPEG=OFF
        -DWITH_1394=OFF
        -DWITH_GTK=OFF
        -DWITH_VFW=OFF
        -DWITH_CUDA=OFF
        -DWITH_OPENEXR=OFF
        -DWITH_OPENMP=OFF
        -DWITH_TBB=OFF
        -DWITH_IPP=ON
        -DPNG_INCLUDE_DIR=${THIRD_PARTY_PREFIX}/include
        -DPNG_LIBRARY=${THIRD_PARTY_PREFIX}/lib/libpng.a
        -DJPEG_INCLUDE_DIR=${THIRD_PARTY_PREFIX}/include
        -DJPEG_LIBRARY=${THIRD_PARTY_PREFIX}/lib/libjpeg.a
        -DOPENCV_IPPICV_URL=http://${HTTP_SERVER}/deps/opencv342-deps/
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_LIBDIR=lib
        -DCMAKE_PREFIX_PATH=${THIRD_PARTY_PREFIX}
        -DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_PREFIX}
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --parallel $ENV{J}
    DEPENDS libjpeg-turbo-202 libpng
)

# rapidxml
ExternalProject_Add(
    rapidxml
    PREFIX rapidxml
    URL http://${HTTP_SERVER}/deps/rapidxml-1.13.zip
    URL_HASH SHA256=c3f0b886374981bb20fabcf323d755db4be6dba42064599481da64a85f5b3571
    CONFIGURE_COMMAND
        rm -rf ${THIRD_PARTY_PREFIX}/include/rapidxml &&
        mkdir -p ${THIRD_PARTY_PREFIX}/include/rapidxml &&
        cp -r ./ ${THIRD_PARTY_PREFIX}/include/rapidxml/
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE 1
)

# gflags
ExternalProject_Add(
    gflags
    PREFIX gflags
    URL http://${HTTP_SERVER}/deps/gflags-v2.2.2.tar.gz
    URL_HASH SHA256=34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf
    BUILD_IN_SOURCE 0
    CMAKE_ARGS
      -DBUILD_SHARED_LIBS=0 -DBUILD_STATIC_LIBS=1
      -DCMAKE_CXX_FLAGS=-fPIC
      -DCMAKE_C_FLAGS=-fPIC
      -DCMAKE_BUILD_TYPE=Release
      -DBUILD_SHARED_LIBS=OFF
      -DBUILD_STATIC_LIBS=ON
      -DCMAKE_INSTALL_LIBDIR=lib
      -DCMAKE_PREFIX_PATH=${THIRD_PARTY_PREFIX}
      -DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_PREFIX}
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --parallel $ENV{J}
)