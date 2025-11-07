# 快速开始指南

## 🚀 三步构建镜像

### 1. 构建镜像

```bash
# 使用构建脚本（推荐）
./build.sh my-project v1.0

# 或手动构建
docker build --platform linux/amd64 \
  -t gcr.io/mde-cloud/image-repo/my-project:python3.13-cu126-v1.0 .
```

### 2. 推送到 GCR

```bash
docker push gcr.io/mde-cloud/image-repo/my-project:python3.13-cu126-v1.0
```

### 3. 在 HPC 集群使用

```yaml
image: gcr.io/mde-cloud/image-repo/my-project:python3.13-cu126-v1.0
```

## 📦 镜像内容

- ✅ Python 3.13.0
- ✅ CUDA 12.6.2
- ✅ cuDNN 9.x
- ✅ Ubuntu 22.04
- ✅ 基础开发工具（git, vim, wget, curl）
- ✅ 编译工具链

## 🔧 后续安装

### 安装 PyTorch

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 安装 vLLM

```bash
VLLM_USE_PRECOMPILED=1 pip3 install vllm --extra-index-url https://download.pytorch.org/whl/cu124
```

### 从源码安装 vLLM（开发版）

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip3 install --editable . --extra-index-url https://download.pytorch.org/whl/cu124
```

## 📖 详细文档

查看 [README.docker.md](README.docker.md) 获取完整文档。

## 🆘 获取帮助

- Teams/Slack: hpc-admins
- 文档: README.docker.md

