FROM dataelement/bisheng-rt-base:0.0.1
MAINTAINER "author, dataelem.com"

ARG PIP_REPO=https://mirrors.tencent.com/pypi/simple
ARG NEXUS_REPO="https://public2:qTongs8YdIwXSRPX@nexus.dataelem.com/repository/product/bisheng"
ARG EXTRA_PIP_REPO="https://public:26rS9HRxDqaVy5T@nx.dataelem.com/repository/pypi-hosted/simple"

# install missing deps
RUN apt update && apt install libarchive-dev patchelf libgl1 libjsoncpp-dev -y

# configure language
RUN locale-gen en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

# configure timezone
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN ln -s /usr/local/bin/pip3 /usr/bin/pip3.8

# install keys deps for bisheng-pybackend-libs 
RUN pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install tensorflow-1.15.5+nv-cp38-cp38-linux_x86_64.whl -i ${EXTR_PIP_REPO} --extra-index-url ${PIP_REPO}

# install flash-attn, slowly, be patiently, about 15-30mins
RUN wget ${NEXUS_REPO}/flash-attention-2.3.3.tar.gz && tar zxf flash-attention-2.3.3.tar.gz && \
    pushd flash-attention-2.3.3 && \
    MAX_JOBS=10 pip install . -i $PIP_REPO && \
    MAX_JOBS=10 pip install csrc/layer_norm -i $PIP_REPO && \
    popd && \
    rm -f flash-attention-2.3.3.tar.gz && \
    rm -fr flash-attention-2.3.3

RUN pip install lanms==1.0.2 -i ${EXTRA_PIP_REPO}

# install full deps for bisheng-pybackend-libs
RUN pip3 install bisheng-pybackend-libs==0.0.1.post1

# clean caches
RUN echo "clean" 
RUN apt-get clean &&  rm -rf /var/lib/apt/lists/* && rm -rf /root/.cache/pip && rm -fr /opt/bisheng-rt/deps
