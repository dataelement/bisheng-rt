cmake_minimum_required (VERSION 3.18)

project(test LANGUAGES C CXX)

set(TRITON_BUILD_INSTALl_DIR "${PROJ_ROOT_DIR}/tritonbuild/install")
set(TRITON_THIRD_PARTY_INSTALL_PREFIX 
  "${PROJ_ROOT_DIR}/tritonbuild/tritonserver/build/third-party" CACHE STRING "")

list(APPEND CMAKE_PREFIX_PATH 
  "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/googletest/lib/cmake"       # gtest
  "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/common/lib/cmake/gflags"    # gflags
  "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/protobuf/lib/cmake"         # protobuf
  "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/absl/lib/cmake"             # absl
  "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/common/share"               # opencv
  "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/nlohmann_json/lib/cmake")   # nlo-json

set(FRAMEWORK_LIB_INSTALL_PREFIX 
  "${PROJ_ROOT_DIR}/tritonbuild/tritonserver/install")
list(APPEND CMAKE_PREFIX_PATH "${FRAMEWORK_LIB_INSTALL_PREFIX}/lib/cmake")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Unit tests
#
find_package(GTest REQUIRED)

find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)

find_package(TritonCommon REQUIRED)
#find_package(TritonCore REQUIRED)
#find_package(TritonBackend REQUIRED)

#
# dependence library
#
set(DATAELEM_ALG_INC
  "${PROJ_ROOT_DIR}/src/backends/dataelem_backend/include")

set(DATAELEM_ALG_LIBS 
    "${PROJ_ROOT_DIR}/tritonbuild/install/backends/dataelem/libdataelem_alg.a"
    "-L${PROJ_ROOT_DIR}/tritonbuild/install/lib -ltritonserver"
)

set(FRAMEWORK_INC
  "${PROJ_ROOT_DIR}/src/common/include/triton/common"
  "${PROJ_ROOT_DIR}/src/backend/include"
  "${PROJ_ROOT_DIR}/src/core/include"
  "${PROJ_ROOT_DIR}/tritonbuild/tritonserver/install/include"
  "${PROJ_ROOT_DIR}/tritonbuild/tritonserver/build/third-party/protobuf/include"
  "${PROJ_ROOT_DIR}/src/server/src/license"
)

set(FRAMEWORK_LIBS
  "${PROJ_ROOT_DIR}/tritonbuild/backends/dataelem/install/lib/libtritonbackendutils.a"
)

# compile tests

set(test_srcs_
  test_cipher.cc
  test_main.cc
  test_matutils.cc
)

set(test_srcs)
foreach(s_ ${test_srcs_})
  list(APPEND test_srcs ${CMAKE_CURRENT_SOURCE_DIR}/common/${s_})
endforeach()

set(test_target ut_all)
add_executable(${test_target} ${test_srcs})

target_include_directories(
  ${test_target}
  PRIVATE
    ${DATAELEM_ALG_INC}
    ${FRAMEWORK_INC})

target_link_libraries(
  ${test_target} 
  PRIVATE
    GTest::gtest
    ${DATAELEM_ALG_LIBS}
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
    absl::strings
    absl::str_format
    TritonCommon::triton-common-cipher
    )

target_link_options(${test_target} PUBLIC "-Wl,--as-needed")


# compile tests

set(tests_srcs
  cc/license/license_test.cc
)

foreach(src ${tests_srcs})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})

  target_include_directories(
    ${name}
    PRIVATE
      ${FRAMEWORK_INC}
      ${DATAELEM_ALG_INC}
      )

  target_link_libraries(
    ${name}
    PRIVATE
      gflags
      TritonCommon::triton-common-cipher
      # proto-library
      ${FRAMEWORK_LIBS}
      protobuf::libprotobuf
      absl::strings           # Absl
      absl::str_format
      )

  target_link_options(${name} PUBLIC "-Wl,--as-needed")
endforeach()
