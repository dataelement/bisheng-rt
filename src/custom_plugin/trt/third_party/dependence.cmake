
set(HTTP_SERVER "192.168.106.8")
set(THIRD_PARTY_PREFIX "./build/third-party" CACHE STRING "third party libraries")

include(ExternalProject)

# cub
ExternalProject_Add(
    cub
    PREFIX cub
    URL http://${HTTP_SERVER}/pkgs/cub-v1.8.0.tar.gz
    URL_HASH SHA256=6a5363bdbfde42423d2383ac87360f337cf307f3b81111811538b1402280b127
    CONFIGURE_COMMAND
        rm -rf ${THIRD_PARTY_PREFIX}/cub &&
        mkdir -p ${THIRD_PARTY_PREFIX}/cub &&
        cp -r ./ ${THIRD_PARTY_PREFIX}/cub/
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE 1
)

# ExternalProject_Add(
#     protobuf
#     PREFIX prtobuf
#     URL http://${HTTP_SERVER}/pkgs/protobuf-3.8.0.tar.gz
#     URL_HASH SHA256=b4d56ff7596589248ef2d46f73b76a32a7d95edafa4ad5f0813419a5f783586b
#     CONFIGURE_COMMAND
#         rm -rf ${THIRD_PARTY_PREFIX}/protobuf &&
#         mkdir -p ${THIRD_PARTY_PREFIX}/protobuf &&
#         cp -r ./ ${THIRD_PARTY_PREFIX}/protobuf/
#     BUILD_COMMAND ""
#     INSTALL_COMMAND ""
#     BUILD_IN_SOURCE 1
# )