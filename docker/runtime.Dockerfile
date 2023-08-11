FROM tritonserver:22.08
MAINTAINER "hanfeng, dataelem.com"

ARG PIP_REPO=https://mirrors.tencent.com/pypi/simple

# 安装系统库依赖
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y nasm zlib1g-dev libssl-dev libre2-dev libb64-dev locales

# Configure language
RUN locale-gen en_US.UTF-8
ENV LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

# Configure timezone
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安装bisheng-rt
RUN mkdir -p /opt/bisheng-rt/
COPY ./install/ /opt/bisheng-rt/
COPY ./deps /opt/bisheng-rt/deps
WORKDIR /opt/bisheng-rt

# 安装bisheng-rt依赖
RUN ln -s /usr/local/bin/pip3 /usr/bin/pip3.8
RUN pip install -r deps/requirements.txt -i $PIP_REPO

RUN cd deps/flash-attention && \
  pip install . -i $PIP_REPO && \
  pip install csrc/layer_norm -i $PIP_REPO && \
  pip install csrc/rotary -i $PIP_REPO

# Clean caches
RUN apt-get clean &&  rm -rf /var/lib/apt/lists/* && rm -rf /root/.cache/pip && rm -fr /opt/bisheng-rt/deps

EXPOSE 7001

CMD [ "./bin/rtserver" "-f" ]