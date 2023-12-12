FROM ubuntu:20.04
MAINTAINER "author, dataelem.com"

# just a mirror copy from nvcr.io/nvidia/tritonserver:22.12-py3-min
ARG PIP_REPO=https://mirrors.tencent.com/pypi/simple

RUN pip download libs-tritonserver-2212-py3-min==0.0.1 -i ${PIP_REPO} && \
 tar zxf libs-tritonserver-2212-py3-min-0.0.1.tar.gz && \
 cp -fr libs-tritonserver-2212-py3-min-0.0.1/lib/share/* /usr/ && \
 rm libs-tritonserver-2212-py3-min-0.0.1.tar.gz && \
 rm -fr libs-tritonserver-2212-py3-min-0.0.1
