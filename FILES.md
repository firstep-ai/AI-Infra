# 📁 文件清单

本目录包含用于构建 Python 3.13 + CUDA 12.6 基础镜像的所有文件。

## 核心文件

| 文件 | 说明 |
|------|------|
| `Dockerfile` | Python 3.13 + CUDA 12.6 镜像定义文件 |
| `build.sh` | 自动化构建脚本（可执行） |
| `.dockerignore` | Docker 构建忽略文件 |

## 文档文件

| 文件 | 说明 |
|------|------|
| `README.docker.md` | 完整的使用文档 |
| `QUICKSTART.md` | 快速开始指南 |
| `FILES.md` | 本文件 - 文件清单 |

## 示例文件

| 文件 | 说明 |
|------|------|
| `example-hpc-job.yaml` | HPC 集群作业配置示例 |

## 📋 文件用途

### Dockerfile
- 定义镜像构建步骤
- 安装 Python 3.13.0（从源码编译）
- 基于 NVIDIA CUDA 12.6.2 + cuDNN 9
- 不预装 PyTorch 和 vLLM（由用户后续安装）

### build.sh
- 自动化构建流程
- 交互式确认
- 自动推送到 GCR（可选）
- 显示后续步骤提示

### .dockerignore
- 优化构建上下文
- 排除不必要的文件
- 减小构建时间

### README.docker.md
- 详细的构建和使用说明
- 安装 PyTorch 和 vLLM 的方法
- 常见问题解答
- 故障排除指南

### QUICKSTART.md
- 快速开始的三步指南
- 常用命令参考
- 简洁明了

### example-hpc-job.yaml
- HPC 集群作业配置模板
- 展示如何使用构建的镜像
- 包含资源配置示例

## 🚀 使用流程

1. **阅读文档**: `QUICKSTART.md` 或 `README.docker.md`
2. **构建镜像**: 运行 `./build.sh my-project v1.0`
3. **推送镜像**: 脚本会提示是否推送
4. **使用镜像**: 参考 `example-hpc-job.yaml`
5. **安装依赖**: 在容器中安装 PyTorch 和 vLLM

## 📊 镜像规格

- **Python**: 3.13.0
- **CUDA**: 12.6.2
- **cuDNN**: 9.x
- **OS**: Ubuntu 22.04
- **大小**: ~5-6 GB
- **构建时间**: ~15-20 分钟

## 💡 特点

- ✅ 使用 Python 3.13 最新版本
- ✅ 支持 CUDA 12.6 最新 GPU
- ✅ 不预装 vLLM，由用户自行安装
- ✅ 轻量级基础镜像
- ✅ 灵活可定制

## 🔗 相关链接

- [Python 3.13 发布说明](https://www.python.org/downloads/release/python-3130/)
- [NVIDIA CUDA 容器](https://hub.docker.com/r/nvidia/cuda)
- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)
- [vLLM 官方文档](https://docs.vllm.ai/)

