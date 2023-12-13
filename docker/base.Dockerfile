FROM nvcr.io/nvidia/tritonserver:22.12-py3-min

ARG NEXUS_REPO="https://public2:qTongs8YdIwXSRPX@nexus.dataelem.com/repository/product/bisheng"

RUN apt update && apt-get -y install language-pack-zh-hans
RUN localedef -c -f UTF-8 -i zh_CN zh_CN.utf8
RUN locale-gen en_US.UTF-8
ENV LC_ALL=zh_CN.utf8

# install system libs
RUN apt install -y nasm zlib1g-dev rapidjson-dev libssl-dev libboost1.71-dev libre2-dev libb64-dev libarchive-dev patchelf

# install cv libs
RUN apt install -y libsm6 libxext6 libxrender-dev libgl1

# install python env
RUN apt install -y python3.8 libpython3.8-dev python3-pip
RUN local repo="https://mirrors.aliyun.com/pypi/simple"
RUN pip3 install --upgrade wheel setuptools -i repo
RUN pip3 install --upgrade numpy -i $repo

# install dcgm
RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get update
RUN apt-get install -y datacenter-gpu-manager
RUN rm cuda-keyring_1.0-1_all.deb

# install cmake
RUN wget ${NEXUS_REPO}/cmake-3.23.1-linux-x86_64.tar.gz
RUN tar zxf cmake-3.23.1-linux-x86_64.tar.gz --strip-components 1 -C /usr/local
RUN rm cmake-3.23.1-linux-x86_64.tar.gz

# Clean caches
RUN echo "clean" 
RUN apt-get clean &&  rm -rf /var/lib/apt/lists/* && rm -rf /root/.cache/pip && rm -fr /opt/bisheng-rt/deps