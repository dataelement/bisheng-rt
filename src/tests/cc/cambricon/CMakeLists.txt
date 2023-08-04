# Copyright 2022, DataElem Inc. All rights reserved.

cmake_minimum_required(VERSION 3.16.3)

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
set(PROJ_ROOT_DIR "/root/workspace/projects/idpserver")
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
  demo
  demo.cc
)

target_include_directories(
  demo
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    "/usr/local/neuware/include/"
)

target_link_libraries(
  demo
  PRIVATE
    ${OpenCV_LIBS}
    "-L /usr/local/neuware/lib64/"
    "-lmagicmind -lmagicmind_runtime -lcnrt"
)