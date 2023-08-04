# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(TFLIB LANGUAGES CXX CUDA)

include(cmake/Modules/set_ifndef.cmake)

set(DEFAULT_CUDA_VERSION 10.0)
set_ifndef(CUDA_VERSION ${DEFAULT_CUDA_VERSION})
message(STATUS "CUDA version set to ${CUDA_VERSION}")

find_package(CUDA ${CUDA_VERSION} REQUIRED)

set(TF_VER "1" CACHE STRING "TF VER")


option(BUILD_TF "Build in TensorFlow mode" OFF)
option(BUILD_TF_CC "Build in TensorFlow nn-predictor mode" OFF)
message(STATUS "BUILD_TF ${BUILD_TF}")
message(STATUS "BUILD_TF_CC ${BUILD_TF_CC}")

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

set(TF_PATH "" CACHE STRING "TensorFlow path")

if(BUILD_TF AND NOT TF_PATH)
  message(FATAL_ERROR "TF_PATH must be set if BUILD_TF(=TensorFlow mode) is on.")
endif()

if(BUILD_TF_CC AND NOT TF_PATH)
  message(FATAL_ERROR "TF_PATH must be set if BUILD_TF(=TensorFlow nn-predictor mode) is on.")
endif()

list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)
find_package(CUDA REQUIRED)

# setting compiler flags
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")

if (SM STREQUAL 80 OR
    SM STREQUAL 86 OR
    SM STREQUAL 70 OR
    SM STREQUAL 75 OR
    SM STREQUAL 61 OR
    SM STREQUAL 60)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${SM},code=\\\"sm_${SM},compute_${SM}\\\" -rdc=true")
  if (SM STREQUAL 70 OR SM STREQUAL 75 OR SM STREQUAL 80 OR SM STREQUAL 86)
    set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
  endif()
  message("-- Assign GPU architecture (sm=${SM})")
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                        -gencode=arch=compute_60,code=\\\"sm_60,compute_60\\\" \
                        -gencode=arch=compute_61,code=\\\"sm_61,compute_61\\\" \
                        -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                        -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                        ")
                        #-rdc=true")
  if (CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
    # Ampere GPU (SM80) support is only available in CUDA versions > 11.0
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                        -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                        -gencode=arch=compute_86,code=\\\"sm_86,compute_86\\\" ")
    message("-- Assign GPU architecture (sm=60,61,70,75,80,86)")
  else()
    message(WARNING "Detected CUDA version is < 11.0. SM80 SM86 not supported.")
    message("-- Assign GPU architecture (sm=60,61,70,75)")
  endif()

  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
endif()

if (CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
  add_definitions(-DCUDA_GREATER_11=1)
endif()

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_STANDARD STREQUAL "14")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++11")
endif()

if(CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
  set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)
else()
  set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUDA_PATH}/include
)

set(COMMON_LIB_DIRS
  ${CUDA_PATH}/lib64
)

if(BUILD_TF)
  list(APPEND COMMON_HEADER_DIRS ${TF_PATH}/include)
  # list(APPEND COMMON_LIB_DIRS ${TF_PATH})
  list(APPEND COMMON_HEADER_DIRS ${PROTOBUF_INC_DIR})
  list(APPEND COMMON_LIB_DIRS ${TF_PATH}/lib)
  set(TF_LIB_DEPS -ltensorflow_framework)
endif()

if(BUILD_TF_CC)
  message(STATUS "Fetching Tensorflow")

  set(TENSORFLOW_CC_DIR ${TF_PATH}/src/tensorflow_cc/tensorflow_cc/build)
  set(PROTOBUF_INCLUDE_DIR ${TENSORFLOW_CC_DIR}/tensorflow/tensorflow/contrib/makefile/gen/protobuf/include/)
  set(PROTOBUF_LIB_DIR ${TENSORFLOW_CC_DIR}/tensorflow/tensorflow/contrib/makefile/gen/protobuf/lib)
  include_directories("${PROTOBUF_INCLUDE_DIR}")

  set(EIGEN3_INCLUDE_DIR ${TENSORFLOW_CC_DIR}/tensorflow/tensorflow/contrib/makefile/downloads/eigen)
  set(ABSL_INCLUDE_DIR ${TENSORFLOW_CC_DIR}/tensorflow/tensorflow/contrib/makefile/downloads/absl)

  set(TF_INC_DIR ${TENSORFLOW_CC_DIR}/tensorflow/ ${TENSORFLOW_CC_DIR}/tensorflow/bazel-genfiles/
                 ${TENSORFLOW_CC_DIR}/tensorflow/tensorflow/contrib/makefile/downloads/nsync/public/)
  set(TF_LIB_DIR ${TENSORFLOW_CC_DIR}/tensorflow/bazel-bin/tensorflow)

  set(TF_LIB_DEPS -ltensorflow_cc ${PROTOBUF_LIB_DIR}/libprotobuf.a)
  message(STATUS "TENSORFLOW lib (${TF_LIB_DEPS}) incr (${TF_INC_DIR})")

  # dependency on Eigen for confusion matrix fast computation
  message(STATUS "EIGEN3_INCLUDE_DIR=(${EIGEN3_INCLUDE_DIR})")
  include_directories("${EIGEN3_INCLUDE_DIR}")
  include_directories("${ABSL_INCLUDE_DIR}")

  list(APPEND COMMON_HEADER_DIRS ${TF_INC_DIR})
  list(APPEND COMMON_LIB_DIRS ${TF_LIB_DIR})
endif()

include_directories(
  ${COMMON_HEADER_DIRS}
)

link_directories(
  ${COMMON_LIB_DIRS}
)

include_directories(plugin)
add_subdirectory(plugin)

