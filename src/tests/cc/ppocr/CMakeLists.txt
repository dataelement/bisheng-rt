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
set(PROJ_ROOT_DIR "/home/public/projects/idpserver")
set(TRITON_SRC_DIR "${PROJ_ROOT_DIR}/src" CACHE STRING "")
set(TRITON_THIRD_PARTY_INSTALL_PREFIX "${PROJ_ROOT_DIR}/tritonbuild/tritonserver/build/third-party" CACHE STRING "")

list(APPEND CMAKE_PREFIX_PATH "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/common/share")
list(APPEND CMAKE_PREFIX_PATH "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/absl/lib/cmake")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---[ Depedences Options
# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "${OpenCV_INCLUDE_DIRS}")
message(STATUS "${OpenCV_LIBS}")

find_package(absl CONFIG REQUIRED)
message(STATUS "ABSL: ${absl_FOUND}")

add_executable(
  ppdet
  ppdet.cc
  ${TRITON_SRC_DIR}/backends/dataelem_backend/src/ext/clipper/clipper.cpp
  ${TRITON_SRC_DIR}/backends/dataelem_backend/src/ext/ppocr/postprocess_op.cpp
)

target_include_directories(
  ppdet
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    "${TRITON_SRC_DIR}/backends/dataelem_backend/include"
    "${TRITON_SRC_DIR}/core/include"
)

target_link_libraries(
  ppdet
  PRIVATE
    ${OpenCV_LIBS}
    absl::strings           # Absl
    absl::str_format
)

add_executable(
  cropconcat
  cropconcat.cc
)

target_include_directories(
  cropconcat
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    "${TRITON_SRC_DIR}/backends/dataelem_backend/include"
    "${TRITON_SRC_DIR}/core/include"
)

target_link_libraries(
  cropconcat
  PRIVATE
    ${OpenCV_LIBS}
    absl::strings           # Absl
    absl::str_format
)

add_executable(
  ppcls
  ppcls.cc
)

target_include_directories(
  ppcls
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    "${TRITON_SRC_DIR}/backends/dataelem_backend/include"
    "${TRITON_SRC_DIR}/core/include"
)

target_link_libraries(
  ppcls
  PRIVATE
    ${OpenCV_LIBS}
    absl::strings           # Absl
    absl::str_format
)

add_executable(
  pprec
  pprec.cc
)

target_include_directories(
  pprec
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    "${TRITON_SRC_DIR}/backends/dataelem_backend/include"
    "${TRITON_SRC_DIR}/core/include"
)

target_link_libraries(
  pprec
  PRIVATE
    ${OpenCV_LIBS}
    absl::strings           # Absl
    absl::str_format
)