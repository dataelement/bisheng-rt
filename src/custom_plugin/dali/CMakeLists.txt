cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_ARCHITECTURES "50;52;60;61;70;75;80;86")

project(custom_dummy_plugin LANGUAGES CUDA CXX C)

# set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(SYSTEM "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

set(TRITON_THIRD_PARTY_INSTALL_PREFIX 
        "${PROJ_ROOT_DIR}/tritonbuild/tritonserver/build/third-party" CACHE STRING "")
list(APPEND CMAKE_PREFIX_PATH "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/common/share")
list(APPEND CMAKE_PREFIX_PATH "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/absl/lib/cmake")

# ---[ Depedences Options
# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "${OpenCV_INCLUDE_DIRS}")
message(STATUS "${OpenCV_LIBS}")

find_package(absl CONFIG REQUIRED)
message(STATUS "ABSL: ${absl_FOUND}")

# Dali
# set(PYTHON_BIN "${PROJ_ROOT_DIR}/tritonbuild/install/backends/dali/conda/envs/dalienv/bin/python3")
set(PYTHON_BIN "/usr/bin/python3")

execute_process(
        COMMAND ${PYTHON_BIN} -c "import nvidia.dali as dali; print(dali.sysconfig.get_lib_dir())"
        OUTPUT_VARIABLE DALI_LIB_DIR)
string(STRIP ${DALI_LIB_DIR} DALI_LIB_DIR)

execute_process(
        COMMAND ${PYTHON_BIN} -c "import nvidia.dali as dali; print(\" \".join(dali.sysconfig.get_compile_flags()))"
        OUTPUT_VARIABLE DALI_COMPILE_FLAGS)
string(STRIP ${DALI_COMPILE_FLAGS} DALI_COMPILE_FLAGS)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DALI_COMPILE_FLAGS} ")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${DALI_COMPILE_FLAGS} ")
link_directories("${DALI_LIB_DIR}")

add_library(customops
  SHARED 
  dummy.cc dummy.cu
  b64dec.cc)

# target_compile_features(customops PUBLIC cxx_std_14)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")

target_link_libraries(
  customops
  PRIVATE
    dali
    # ${OpenCV_LIBS}
    absl::strings
    absl::str_format)
