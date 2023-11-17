FROM dataelement/bisheng-rt-runtime:0.0.1

ARG PIP_REPO=https://mirrors.tencent.com/pypi/simple
ARG BISHENG_PYBACKEND_LIBS_VER=0.0.1
ARG EXTR_PIP_REPO="http://public:26rS9HRxDqaVy5T@110.16.193.170:50083/repository/pypi-hosted/simple --trusted-host 110.16.193.170"

# Configure language
RUN locale-gen en_US.UTF-8
ENV LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

# Configure timezone
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy bins and configs
COPY ./install /opt/bisheng-rt
WORKDIR /opt/bisheng-rt

# private lanms version
RUN pip3 install lanms==1.0.2 -i ${EXTR_PIP_REPO}

RUN pip3 install bisheng-pybackend-libs==${BISHENG_PYBACKEND_LIBS_VER} -i ${PIP_REPO}

RUN apt-get clean &&  rm -rf /var/lib/apt/lists/* && rm -rf /root/.cache/pip

CMD ["bash", "bin/entrypoint.sh"]
