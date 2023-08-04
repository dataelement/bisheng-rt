# Copyright 2022, DataElem Inc. All rights reserved.

cmake_minimum_required(VERSION 3.17)

project(tritondataelembackend LANGUAGES C CXX)

#
# Options
#
# Must include options required for this project as well as any
# projects included in this one by FetchContent.
#
# GPU support is disabled by default because dataelem backend doesn't
# support GPUs.
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" OFF)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

option(ENABLE_PRIVATE_USE "Enable private use in backend" OFF)

set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")

set(PROJ_ROOT_DIR "/home/hanfeng/projects/idpserver" CACHE STRING "")
set(TRITON_SRC_DIR "${PROJ_ROOT_DIR}/src" CACHE STRING "")
set(TRITON_THIRD_PARTY_INSTALL_PREFIX "${PROJ_ROOT_DIR}/tritonbuild/tritonserver/build/third-party" CACHE STRING "")

list(APPEND CMAKE_PREFIX_PATH "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/common/share")
list(APPEND CMAKE_PREFIX_PATH "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/absl/lib/cmake")
list(APPEND CMAKE_PREFIX_PATH "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/nlohmann_json/lib/cmake")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---[ Depedences Options
# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "${OpenCV_INCLUDE_DIRS}")
message(STATUS "${OpenCV_LIBS}")

# nlohmann_json
find_package(nlohmann_json REQUIRED)

find_package(absl CONFIG REQUIRED)
message(STATUS "ABSL: ${absl_FOUND}")
#
# Dependencies
#
# FetchContent's composibility isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

# FetchContent_Declare(
#   repo-common
#   GIT_REPOSITORY https://github.com/triton-inference-server/common.git
#   GIT_TAG ${TRITON_COMMON_REPO_TAG}
#   GIT_SHALLOW ON
# )
# FetchContent_Declare(
#   repo-core
#   GIT_REPOSITORY https://github.com/triton-inference-server/core.git
#   GIT_TAG ${TRITON_CORE_REPO_TAG}
#   GIT_SHALLOW ON
# )
# FetchContent_Declare(
#   repo-backend
#   GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
#   GIT_TAG ${TRITON_BACKEND_REPO_TAG}
#   GIT_SHALLOW ON
# )

FetchContent_Declare(
  repo-common
  SOURCE_DIR ${TRITON_SRC_DIR}/common
)
FetchContent_Declare(
  repo-core
  SOURCE_DIR ${TRITON_SRC_DIR}/core
)
FetchContent_Declare(
  repo-backend
  SOURCE_DIR ${TRITON_SRC_DIR}/backend
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# Shared library implementing the Triton Backend API
#
configure_file(src/libtriton_dataelem.ldscript libtriton_dataelem.ldscript COPYONLY)

set(ALG_SRCS_
  ext/clipper/clipper.cpp
  ext/ppocr/preprocess_op.cpp
  ext/ppocr/postprocess_op.cpp
  ext/ppocr/utility.cpp
  dataelem/framework/alg.cc
  dataelem/framework/alg_utils.cc
  dataelem/common/mat_utils.cc
  dataelem/common/json_utils.cc
  dataelem/alg/eg_alg.cc
  dataelem/alg/lanms.cc
  #dataelem/alg/dbnet.cc
  dataelem/alg/imgdecode.cc
  dataelem/alg/cropconcat.cc
  dataelem/alg/ppdet.cc
  dataelem/alg/ppcls.cc
  dataelem/alg/pprec.cc
  dataelem/alg/general_prep.cc
  dataelem/alg/mrcnn_v5.cc
  dataelem/alg/recog_helper.cc
  dataelem/alg/transformer.cc
  dataelem/alg/ocr_app.cc
  dataelem/alg/distribution.cc
  dataelem/common/apidata.cc
  ext/rmustache/mustache.cc
  dataelem/framework/app.cc
  dataelem/apps/bls_app.cc
  dataelem/apps/ocr_app.cc
  dataelem/alg/east.cc
  dataelem/alg/mrcnn.cc
  dataelem/alg/transformer_v1.cc
  dataelem/alg/two_classification.cc
  dataelem/alg/tablemrcnn.cc
  dataelem/alg/crnn.cc
  )

set(ALG_SRCS)
foreach(s_ ${ALG_SRCS_})
  list(APPEND ALG_SRCS src/${s_})
endforeach()

add_library(
  triton-dataelem-backend SHARED
  src/dataelem/backend/backend.cc
  ${ALG_SRCS}
)

add_library(
  TritonDataelemBackend::triton-dataelem-backend ALIAS triton-dataelem-backend
)

target_include_directories(
  triton-dataelem-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(triton-dataelem-backend PRIVATE cxx_std_14)
target_compile_options(
  triton-dataelem-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Werror>
)

if(${TRITON_ENABLE_GPU})
target_compile_definitions(
  triton-dataelem-backend
  PUBLIC TRITON_ENABLE_GPU=1
)
endif() # TRITON_ENABLE_GPU

target_link_libraries(
  triton-dataelem-backend
  PRIVATE
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
    triton-common-error     # from repo-common
    triton-common-fs        # from repo-common
    ${OpenCV_LIBS}          # OpenCV
    nlohmann_json::nlohmann_json
    absl::strings           # Absl
    absl::str_format
)

set_target_properties(
  triton-dataelem-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME triton_dataelem
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_dataelem.ldscript
  LINK_FLAGS "-Wl,--version-script libtriton_dataelem.ldscript"
)

# static library
add_library(
  dataelem-alg-library
  ${ALG_SRCS}
)

if(${TRITON_ENABLE_GPU})
target_compile_definitions(
  dataelem-alg-library
  PUBLIC TRITON_ENABLE_GPU=1
)
endif() # TRITON_ENABLE_GPU

target_compile_features(dataelem-alg-library PRIVATE cxx_std_14)
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(
    dataelem-alg-library
    PRIVATE
      /W1 /D_WIN32_WINNT=0x0A00 /EHsc
  )
else()
  target_compile_options(
    dataelem-alg-library
    PRIVATE
      -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-declarations -Werror
  )
endif()

set_target_properties(
  dataelem-alg-library
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME dataelem_alg
)

target_include_directories(
  dataelem-alg-library
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(
  dataelem-alg-library
  PUBLIC
    triton-core-serverstub          # from repo-core
    triton-backend-utils            # from repo-backend
    ${OpenCV_LIBS}                  # OpenCV
    nlohmann_json::nlohmann_json    # json
    absl::strings                   # Absl
    absl::str_format
)

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonDataelemBackend)

install(
  TARGETS
    triton-dataelem-backend
    dataelem-alg-library
  EXPORT
    triton-dataelem-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/dataelem
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/dataelem
)

install(
  EXPORT
    triton-dataelem-backend-targets
  FILE
    TritonDataelemBackendTargets.cmake
  NAMESPACE
    TritonDataelemBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonDataelemBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonDataelemBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonDataelemBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-dataelem-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonDataelemBackendTargets.cmake
  NAMESPACE TritonDataelemBackend::
)

export(PACKAGE TritonDataelemBackend)
