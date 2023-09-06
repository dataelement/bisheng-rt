cmake_minimum_required(VERSION 3.17)

project(trironpaddlebackend LANGUAGES C CXX)

option(TRITON_ENABLE_GPU "Enable GPU support in backend" OFF)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

set(ENFLAME_INFERENCE_DIR "" CACHE PATH "Paths to Enflame Inference Directory")
set(ENFLAME_INCLUDE_PATHS "${ENFLAME_INFERENCE_DIR}/include" CACHE PATH "Paths to enflame Inference includes.")
set(ENFLAME_LIB_PATHS "${ENFLAME_INFERENCE_DIR}/lib" CACHE PATH "Paths to enflame Inference libraries.")
set(ENFLAME_LIB_NAME "TopsInference")

set(PROJ_ROOT_DIR "/home/hanfeng/projects/idpserver" CACHE PATH "")
set(TRITON_SRC_DIR "${PROJ_ROOT_DIR}/src" CACHE STRING "")

set(FRAMEWORK_LIB_INSTALL_PREFIX "${PROJ_ROOT_DIR}/tritonbuild/tritonserver/install")
list(APPEND CMAKE_PREFIX_PATH "${FRAMEWORK_LIB_INSTALL_PREFIX}/lib/cmake")

set(TRITON_THIRD_PARTY_INSTALL_PREFIX 
    "${PROJ_ROOT_DIR}/tritonbuild/tritonserver/build/third-party" CACHE STRING "")
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

configure_file(src/libtriton_enflame.ldscript libtriton_enflame.ldscript COPYONLY)

add_library(
  triton-enflame-backend SHARED
  src/enflame.cc
  src/enflame_backend_utils.cc
  ${ENFLAME_INCLUDE_PATHS}/TopsInference/utils/tops_utils.cpp
)

target_include_directories(
  triton-enflame-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(
  triton-enflame-backend
  PRIVATE 
    ${ENFLAME_INCLUDE_PATHS}
    ${ENFLAME_INCLUDE_PATHS}/TopsInference/dtu
    ${ENFLAME_INCLUDE_PATHS}/TopsInference/dtu/libprofile/
    ${ENFLAME_INCLUDE_PATHS}/TopsInference/dtu/logging/
)

target_link_libraries(
  triton-enflame-backend
  PRIVATE "-L${ENFLAME_LIB_PATHS} -l${ENFLAME_LIB_NAME}"
)

target_compile_features(triton-enflame-backend PRIVATE cxx_std_17)
target_compile_options(
  triton-enflame-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
)

set_target_properties(
  triton-enflame-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME triton_enflame
  SKIP_BUILD_RPATH TRUE
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_enflame.ldscript
  LINK_FLAGS "-Wl,--version-script libtriton_enflame.ldscript"
)

target_link_libraries(
  triton-enflame-backend
  PRIVATE
    TritonBackend::triton-backend-utils
    TritonCore::triton-core-serverstub
)
