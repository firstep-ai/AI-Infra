# Python 3.13 + CUDA 12.6 基础镜像
# 适用于后续安装 PyTorch GPU 版本和 vLLM

FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update --yes \
    && apt-get install --yes --no-install-recommends \
        software-properties-common \
        build-essential \
        ca-certificates \
        curl \
        git \
        wget \
        vim \
        libssl-dev \
        libffi-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        liblzma-dev \
        zlib1g-dev \
    && apt-get clean --yes \
    && rm -rf /var/lib/apt/lists/*

# 下载并编译安装 Python 3.13
RUN cd /tmp \
    && wget --no-check-certificate https://www.python.org/ftp/python/3.13.0/Python-3.13.0.tgz \
    && tar -xzf Python-3.13.0.tgz \
    && cd Python-3.13.0 \
    && ./configure --enable-optimizations --with-ensurepip=install \
    && make -j$(nproc) \
    && make altinstall \
    && cd / \
    && rm -rf /tmp/Python-3.13.0*

# 设置 Python 3.13 为默认 python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.13 1 \
    && update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.13 1

# 升级 pip 和安装基础工具
RUN pip3 install --upgrade --no-cache-dir \
    pip \
    setuptools \
    wheel

# 设置工作目录
WORKDIR /workspace

# 验证安装
RUN python3 --version \
    && pip3 --version \
    && nvcc --version

CMD ["/bin/bash"]

