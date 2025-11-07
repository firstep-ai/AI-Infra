# 🎯 从这里开始

> Python 3.13 + CUDA 12.6 Docker 镜像构建

## 📖 我该看哪个文档？

### 🚀 我想快速开始
👉 查看 [QUICKSTART.md](QUICKSTART.md)
- 三步构建镜像
- 常用命令
- 简洁明了

### 📘 我想了解完整信息
👉 查看 [README.docker.md](README.docker.md)
- 详细的构建说明
- 安装 PyTorch 和 vLLM
- 常见问题解答
- 故障排除

### 📊 我想了解总体情况
👉 查看 [DOCKER-IMAGE-GUIDE.md](DOCKER-IMAGE-GUIDE.md)
- 镜像特点和规格
- 使用场景
- 工作流程
- 最佳实践

### 📁 我想知道有哪些文件
👉 查看 [FILES.md](FILES.md)
- 文件清单
- 文件用途说明
- 使用流程

## ⚡ 快速命令

```bash
# 构建镜像
./build.sh my-project v1.0

# 测试镜像
docker run --gpus all -it --rm \
  gcr.io/mde-cloud/image-repo/my-project:python3.13-cu126-v1.0 \
  /bin/bash

# 在容器中安装 PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 在容器中安装 vLLM
VLLM_USE_PRECOMPILED=1 pip3 install vllm --extra-index-url https://download.pytorch.org/whl/cu124
```

## 📦 镜像信息

- **Python**: 3.13.0
- **CUDA**: 12.6.2
- **大小**: ~5-6 GB
- **特点**: 基础镜像，不预装 vLLM

## 📚 所有文档

| 文档 | 说明 |
|------|------|
| [START-HERE.md](START-HERE.md) | 本文件 - 快速导航 |
| [QUICKSTART.md](QUICKSTART.md) | 快速开始指南 |
| [DOCKER-IMAGE-GUIDE.md](DOCKER-IMAGE-GUIDE.md) | 总览指南 |
| [README.docker.md](README.docker.md) | 完整文档 |
| [FILES.md](FILES.md) | 文件清单 |
| [example-hpc-job.yaml](example-hpc-job.yaml) | HPC 作业配置示例 |

## 🔧 核心文件

- `Dockerfile` - 镜像定义
- `build.sh` - 构建脚本（可执行）
- `.dockerignore` - 构建优化

## 💡 推荐阅读路径

### 新手用户
1. START-HERE.md (本文件)
2. QUICKSTART.md
3. 运行 `./build.sh`
4. 查看 README.docker.md 了解更多

### 有经验用户
1. DOCKER-IMAGE-GUIDE.md
2. 直接运行 `./build.sh`
3. 根据需要参考 README.docker.md

## 🆘 需要帮助？

- **快速问题**: 查看 QUICKSTART.md
- **详细问题**: 查看 README.docker.md
- **技术支持**: Teams/Slack 联系 hpc-admins

---

**下一步**: 选择上面的一个文档开始阅读，或直接运行 `./build.sh my-project v1.0`

