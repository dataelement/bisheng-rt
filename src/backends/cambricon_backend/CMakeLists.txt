cmake_minimum_required(VERSION 3.17)

project(trironpaddlebackend LANGUAGES C CXX)

option(TRITON_ENABLE_GPU "Enable GPU support in backend" OFF)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

set(PROJ_ROOT_DIR "/home/hanfeng/projects/idpserver" CACHE PATH "")
set(TRITON_SRC_DIR "${PROJ_ROOT_DIR}/src" CACHE STRING "")

set(CAMBRICON_INFERENCE_DIR "${PROJ_ROOT_DIR}/tritonbuild/third_party/libcambricon_v0.1/" CACHE PATH "Paths to Cambricon Inference Directory")
set(CAMBRICON_INCLUDE_PATHS "${CAMBRICON_INFERENCE_DIR}/include" CACHE PATH "Paths to cambricon Inference includes.")
set(CAMBRICON_LIB_PATHS "${CAMBRICON_INFERENCE_DIR}/lib" CACHE PATH "Paths to cambricon Inference libraries.")

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

configure_file(src/libtriton_cambricon.ldscript libtriton_cambricon.ldscript COPYONLY)

add_library(
  triton-cambricon-backend SHARED
  src/cambricon.cc
  src/cambricon_backend_utils.cc
)

target_include_directories(
  triton-cambricon-backend
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CAMBRICON_INCLUDE_PATHS}
)

target_link_libraries(
  triton-cambricon-backend
  PRIVATE "-L${CAMBRICON_LIB_PATHS} -lmagicmind -lmagicmind_runtime -lcnrt"
)

target_compile_features(triton-cambricon-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-cambricon-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
)

set_target_properties(
  triton-cambricon-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME triton_cambricon
  SKIP_BUILD_RPATH TRUE
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_cambricon.ldscript
  LINK_FLAGS "-Wl,--version-script libtriton_cambricon.ldscript"
)

target_link_libraries(
  triton-cambricon-backend
  PRIVATE
    TritonBackend::triton-backend-utils
    TritonCore::triton-core-serverstub
)
