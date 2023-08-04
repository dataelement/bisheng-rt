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
set(PROJ_ROOT_DIR "/home/liuqingjie/projects/idpserver")
set(TRITON_SRC_DIR "${PROJ_ROOT_DIR}/src" CACHE STRING "")
set(TRITON_THIRD_PARTY_INSTALL_PREFIX "${PROJ_ROOT_DIR}/tritonbuild/tritonserver/build/third-party" CACHE STRING "")

list(APPEND CMAKE_PREFIX_PATH "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/common/share")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---[ Depedences Options
# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "${OpenCV_INCLUDE_DIRS}")
message(STATUS "${OpenCV_LIBS}")

add_executable(
  openvino_test
  openvino_test.cpp
)
target_include_directories(
  openvino_test
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    "/home/liuqingjie/projects/idpserver/tritonbuild/third_party/libopenvino_v22.08_2022.2/include/"
)


target_link_libraries(
  openvino_test
  PRIVATE
    ${OpenCV_LIBS}
    "-L/home/liuqingjie/projects/idpserver/tritonbuild/third_party/libopenvino_v22.08_2022.2/lib -lopenvino"
  )