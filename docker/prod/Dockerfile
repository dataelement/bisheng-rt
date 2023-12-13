FROM dataelement/bisheng-rt-runtime:0.0.1

ARG RT_VER=0.0.5
ARG RT_ENT_VER=0.0.5
ARG BISHENG_PYBACKEND_LIBS_VER=0.0.1.post1

ARG PIP_REPO=https://mirrors.tencent.com/pypi/simple
ARG EXTRA_PIP_REPO="https://public:26rS9HRxDqaVy5T@nx.dataelem.com/repository/pypi-hosted/simple"
ARG NEXUS_REPO="https://public2:qTongs8YdIwXSRPX@nexus.dataelem.com/repository/product/bisheng"

# Configure language
RUN locale-gen en_US.UTF-8
ENV LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

# Configure timezone
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install bisheng-rt bins and configs
# COPY ./install /opt/bisheng-rt
# WORKDIR /opt/bisheng-rt

RUN mkdir -p /opt/bisheng-rt
WORKDIR /opt/bisheng-rt
RUN wget -q ${NEXUS_REPO}/bisheng-rt-${RT_VER}.tar.gz && \
    tar zxf bisheng-rt-${RT_VER}.tar.gz --strip-components 1 -C /opt/bisheng-rt && \
    rm bisheng-rt-${RT_VER}.tar.gz

RUN wget -q ${NEXUS_REPO}/bisheng-rt-${RT_ENT_VER}.tar.gz && \
    tar zxf bisheng-rt-${RT_ENT_VER}.tar.gz --strip-components 1 -C /opt/bisheng-rt && \
    rm bisheng-rt-${RT_ENT_VER}.tar.gz

# private lanms version
RUN pip3 install lanms==1.0.2 -i ${EXTRA_PIP_REPO}
RUN pip3 install bisheng-pybackend-libs==${BISHENG_PYBACKEND_LIBS_VER}

RUN apt-get clean &&  rm -rf /var/lib/apt/lists/* && rm -rf /root/.cache/pip

CMD ["bash", "bin/entrypoint.sh"]
