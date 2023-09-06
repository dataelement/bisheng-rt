cmake_minimum_required(VERSION 3.17)

project(trironpaddlebackend LANGUAGES C CXX)

option(TRITON_ENABLE_GPU "Enable GPU support in backend" OFF)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

set(PROJ_ROOT_DIR "/home/hanfeng/projects/idpserver" CACHE PATH "")
set(TRITON_SRC_DIR "${PROJ_ROOT_DIR}/src" CACHE STRING "")

set(HUAWEI_INFERENCE_DIR "${PROJ_ROOT_DIR}/tritonbuild/third_party/libhuawei_v0.1/" CACHE PATH "Paths to Huawei Inference Directory")
set(HUAWEI_INCLUDE_PATHS "${HUAWEI_INFERENCE_DIR}/include" CACHE PATH "Paths to Huawei Inference includes.")
set(HUAWEI_LIB_PATHS "${HUAWEI_INFERENCE_DIR}/lib" CACHE PATH "Paths to Huawei Inference libraries.")

set(FRAMEWORK_LIB_INSTALL_PREFIX "${PROJ_ROOT_DIR}/tritonbuild/tritonserver/install")
list(APPEND CMAKE_PREFIX_PATH "${FRAMEWORK_LIB_INSTALL_PREFIX}/lib/cmake")

set(TRITON_THIRD_PARTY_INSTALL_PREFIX "${PROJ_ROOT_DIR}/tritonbuild/tritonserver/build/third-party" CACHE STRING "")
list(APPEND CMAKE_PREFIX_PATH "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/common/share")
list(APPEND CMAKE_PREFIX_PATH "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/absl/lib/cmake")

# ---[ Depedences Options
# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OPENCV_INC: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OPENCV_LIB: ${OpenCV_LIBS}")

find_package(absl CONFIG REQUIRED)
message(STATUS "ABSL: ${absl_FOUND}")

find_package(TritonCommon REQUIRED)
find_package(TritonCore REQUIRED)
find_package(TritonBackend REQUIRED)

configure_file(src/libtriton_huawei.ldscript libtriton_huawei.ldscript COPYONLY)

add_library(
  triton-huawei-backend SHARED
  src/huawei.cc
  src/huawei_backend_utils.cc
)

target_include_directories(
  triton-huawei-backend
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${HUAWEI_INCLUDE_PATHS}
)

target_link_libraries(
  triton-huawei-backend
  PRIVATE "-L${HUAWEI_LIB_PATHS} -lascendcl"
)

target_compile_features(triton-huawei-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-huawei-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
)

set_target_properties(
  triton-huawei-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME triton_huawei
  SKIP_BUILD_RPATH TRUE
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_huawei.ldscript
  LINK_FLAGS "-Wl,--version-script libtriton_huawei.ldscript"
)

target_link_libraries(
  triton-huawei-backend
  PRIVATE
    TritonBackend::triton-backend-utils
    TritonCore::triton-core-serverstub
)
