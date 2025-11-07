# Python 3.13 + CUDA 12.6 容器镜像

## 镜像说明

这是一个基础 GPU 容器镜像，包含：
- **Python 3.13.0** (从源码编译)
- **CUDA 12.6.2**
- **cuDNN 9.x**
- **Ubuntu 22.04**

适用于后续安装 PyTorch GPU 版本和 vLLM。

## 构建镜像

### 方法 1: 使用 AIP Job（推荐）

根据 HPC 文档，使用 AIP container image build 功能来构建镜像。

### 方法 2: 本地构建

```bash
# 构建镜像
docker build --platform linux/amd64 \
  -t gcr.io/mde-cloud/image-repo/your-project-name:python3.13-cu126 .

# 推送到 GCR
docker push gcr.io/mde-cloud/image-repo/your-project-name:python3.13-cu126
```

**注意**: 请将 `your-project-name` 替换为您的实际项目名称。

## 使用镜像

### 在 HPC 集群中使用

```yaml
image: gcr.io/mde-cloud/image-repo/your-project-name:python3.13-cu126
```

### 安装 PyTorch

进入容器后，执行以下命令安装 PyTorch：

```bash
# 安装 PyTorch 2.5.x + CUDA 12.4（兼容 CUDA 12.6）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 安装 vLLM

```bash
# 使用预编译版本（推荐）
VLLM_USE_PRECOMPILED=1 pip3 install vllm --extra-index-url https://download.pytorch.org/whl/cu124

# 或者从源码编译安装
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip3 install --editable . --extra-index-url https://download.pytorch.org/whl/cu124
```

### 验证安装

```python
import torch
print(f"Python version: {__import__('sys').version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
```

## 技术规格

| 项目 | 版本/信息 |
|------|----------|
| 基础镜像 | nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04 |
| Python | 3.13.0 |
| CUDA | 12.6.2 |
| cuDNN | 9.x |
| 操作系统 | Ubuntu 22.04 |
| 架构 | linux/amd64 |

## 兼容性

### HPC 集群
- ✅ HPC-2
- ✅ HPC-3
- ✅ HPC-4

### PyTorch 版本
CUDA 12.6 支持以下 PyTorch 版本：
- PyTorch 2.4.x（使用 cu124）
- PyTorch 2.5.x（使用 cu124）✅ 推荐

### CUDA 版本说明
- CUDA 12.6 向后兼容 cu124 的 PyTorch wheels
- PyTorch 官方使用 `cu124` 标识 CUDA 12.4+ 版本

## 常见问题

### Q1: 为什么选择 CUDA 12.6？

CUDA 12.6 是较新的稳定版本，具有以下优势：
- 支持最新的 GPU 架构
- 兼容 PyTorch 2.5+ 和 vLLM 最新版本
- 性能优化和 bug 修复

### Q2: Python 3.13 的兼容性如何？

Python 3.13 是最新的稳定版本。如果遇到兼容性问题，可以修改 Dockerfile 降级到 Python 3.12：

```dockerfile
# 修改下载链接为 Python 3.12
RUN cd /tmp \
    && wget https://www.python.org/ftp/python/3.12.7/Python-3.12.7.tgz \
    && tar -xzf Python-3.12.7.tgz \
    && cd Python-3.12.7 \
    && ./configure --enable-optimizations --with-ensurepip=install \
    && make -j$(nproc) \
    && make altinstall \
    && cd / \
    && rm -rf /tmp/Python-3.12.7*

# 设置 Python 3.12 为默认
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.12 1
```

### Q3: 如何使用不同的 CUDA 版本？

修改 Dockerfile 第一行的基础镜像：

```dockerfile
# CUDA 11.8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# CUDA 12.4
FROM nvidia/cuda:12.4.1-cudnn9-devel-ubuntu22.04
```

相应地，安装 PyTorch 时也需要匹配 CUDA 版本：

```bash
# CUDA 11.8
pip3 install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.4+
pip3 install torch --index-url https://download.pytorch.org/whl/cu124
```

## 扩展镜像

如果您需要预装其他依赖，可以创建一个新的 Dockerfile 扩展这个镜像：

```dockerfile
FROM gcr.io/mde-cloud/image-repo/your-project-name:python3.13-cu126

# 安装 PyTorch
RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# 安装其他依赖
RUN pip3 install --no-cache-dir \
    transformers \
    accelerate \
    datasets \
    numpy \
    pandas

WORKDIR /workspace
```

## 获取帮助

如有问题，请联系：
- Teams/Slack: hpc-admins
- 或参考 HPC 文档

## 参考链接

- [Python 官方下载](https://www.python.org/downloads/)
- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA 容器镜像](https://hub.docker.com/r/nvidia/cuda)
- [vLLM 官方文档](https://docs.vllm.ai/)

