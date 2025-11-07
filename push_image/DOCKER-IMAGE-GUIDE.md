# 🐳 Python 3.13 + CUDA 12.6 Docker 镜像指南

> 为 Rakuten HPC 集群构建的基础 GPU 容器镜像

## ✨ 镜像特点

- ✅ **Python 3.13.0** - 最新稳定版本
- ✅ **CUDA 12.6.2** - 支持最新 GPU 架构
- ✅ **cuDNN 9.x** - 深度学习加速
- ✅ **Ubuntu 22.04** - 稳定的基础系统
- ✅ **轻量级** - 仅包含基础环境（~5-6 GB）
- ✅ **灵活** - 不预装 vLLM，由用户自行安装

## 🚀 快速开始

### 一键构建

```bash
./build.sh my-project v1.0
```

### 手动构建

```bash
docker build --platform linux/amd64 \
  -t gcr.io/mde-cloud/image-repo/my-project:python3.13-cu126-v1.0 .

docker push gcr.io/mde-cloud/image-repo/my-project:python3.13-cu126-v1.0
```

### 在 HPC 使用

```yaml
image: gcr.io/mde-cloud/image-repo/my-project:python3.13-cu126-v1.0
```

## 📦 后续安装

### 1. 安装 PyTorch

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. 安装 vLLM

```bash
# 预编译版本（推荐）
VLLM_USE_PRECOMPILED=1 pip3 install vllm --extra-index-url https://download.pytorch.org/whl/cu124

# 从源码安装（开发版）
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip3 install --editable . --extra-index-url https://download.pytorch.org/whl/cu124
```

## 📁 文件结构

```
AI-Infra/
├── Dockerfile                  # 镜像定义文件
├── build.sh                    # 自动化构建脚本 ⭐
├── .dockerignore               # 构建优化配置
│
├── DOCKER-IMAGE-GUIDE.md       # 本文件 - 总览指南
├── README.docker.md            # 完整文档
├── QUICKSTART.md               # 快速开始
├── FILES.md                    # 文件清单
└── example-hpc-job.yaml        # HPC 作业配置示例
```

## 📖 文档导航

| 文档 | 用途 |
|------|------|
| **DOCKER-IMAGE-GUIDE.md** | 总览指南（本文件） |
| **QUICKSTART.md** | 快速开始，三步上手 |
| **README.docker.md** | 完整文档，详细说明 |
| **FILES.md** | 文件清单和说明 |

## 🎯 使用场景

### 场景 1: 开发和实验
- 使用基础镜像
- 灵活安装不同版本的 PyTorch 和 vLLM
- 快速迭代和测试

### 场景 2: 生产部署
- 基于此镜像创建扩展镜像
- 预装固定版本的依赖
- 稳定可靠

### 场景 3: vLLM 开发
- 从源码安装 vLLM
- 修改和调试 vLLM 代码
- 贡献到 vLLM 项目

## 🔧 技术规格

| 项目 | 版本/信息 |
|------|----------|
| **基础镜像** | nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04 |
| **Python** | 3.13.0 (从源码编译) |
| **CUDA** | 12.6.2 |
| **cuDNN** | 9.x |
| **操作系统** | Ubuntu 22.04 LTS |
| **架构** | linux/amd64 |
| **镜像大小** | ~5-6 GB |
| **构建时间** | ~15-20 分钟 |

## ✅ 兼容性

### HPC 集群
- ✅ HPC-2
- ✅ HPC-3
- ✅ HPC-4

### PyTorch 版本
- ✅ PyTorch 2.4.x (cu124)
- ✅ PyTorch 2.5.x (cu124) - 推荐

### GPU 架构
- ✅ Ampere (A100, A40, A30)
- ✅ Hopper (H100, H200)
- ✅ Ada Lovelace (RTX 4090, L40S)

## 💡 最佳实践

1. **使用构建脚本**: `./build.sh` 简化构建流程
2. **版本标签**: 使用语义化版本（如 v1.0, v1.1）
3. **测试验证**: 构建后先本地测试再推送
4. **文档记录**: 记录使用的镜像版本和配置
5. **依赖管理**: 在容器中使用 requirements.txt 管理 Python 依赖

## ⚠️ 重要提示

1. **Python 3.13**: 最新版本，某些库可能还未完全支持
2. **CUDA 版本**: 使用 CUDA 12.6，兼容 cu124 的 PyTorch wheels
3. **不预装 vLLM**: 由用户根据需求自行安装
4. **镜像前缀**: 必须使用 `gcr.io/mde-cloud/image-repo/`
5. **权限要求**: 推送到 GCR 需要相应权限

## 🔄 常见工作流程

### 工作流程 1: 基础使用

```bash
# 1. 构建镜像
./build.sh my-project v1.0

# 2. 在 HPC 集群使用
# 在作业配置中指定镜像

# 3. 在容器中安装依赖
pip3 install torch vllm transformers
```

### 工作流程 2: 创建扩展镜像

```dockerfile
# Dockerfile.extended
FROM gcr.io/mde-cloud/image-repo/my-project:python3.13-cu126-v1.0

# 安装固定版本的依赖
RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

RUN VLLM_USE_PRECOMPILED=1 pip3 install --no-cache-dir \
    vllm==0.6.3 \
    --extra-index-url https://download.pytorch.org/whl/cu124

WORKDIR /workspace
```

```bash
# 构建扩展镜像
docker build -f Dockerfile.extended \
  -t gcr.io/mde-cloud/image-repo/my-project:vllm-v1.0 .
```

## 🆘 获取帮助

### 问题排查

1. **构建失败**: 检查 Docker 日志，确认网络连接
2. **推送失败**: 确认 GCR 权限，联系 hpc-admins
3. **运行错误**: 查看容器日志，检查 CUDA 版本
4. **兼容性问题**: 参考 README.docker.md 的常见问题

### 联系支持

- **Teams/Slack**: hpc-admins
- **HPC 文档**: 内部 HPC 容器镜像文档
- **vLLM 问题**: https://github.com/vllm-project/vllm/issues

## 📚 参考资源

### 官方文档
- [Python 3.13 发布说明](https://www.python.org/downloads/release/python-3130/)
- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)
- [vLLM 官方文档](https://docs.vllm.ai/)
- [NVIDIA CUDA 容器](https://hub.docker.com/r/nvidia/cuda)

### Rakuten 内部
- [HPC 容器镜像仓库](https://ghe.rakuten-it.com/distributed-training-section/hpc-container-images)
- HPC 集群使用文档

## 📝 更新日志

### v1.0 (2025-11-06)
- ✅ 初始版本
- ✅ Python 3.13.0
- ✅ CUDA 12.6.2
- ✅ 基础镜像，不预装 vLLM
- ✅ 完整文档和构建脚本

---

**下一步**: 
1. 阅读 [QUICKSTART.md](QUICKSTART.md) 快速上手
2. 运行 `./build.sh my-project v1.0` 构建镜像
3. 查看 [README.docker.md](README.docker.md) 了解详细信息

**提示**: 将 `my-project` 替换为您的实际项目名称

