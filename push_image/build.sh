#!/bin/bash

# Python 3.13 + CUDA 12.6 镜像构建脚本
# 用法: ./build.sh [项目名称] [标签]

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 检查参数
if [ $# -lt 1 ]; then
    echo -e "${RED}用法: $0 [项目名称] [标签]${NC}"
    echo ""
    echo "示例:"
    echo "  $0 my-project v1.0"
    echo "  $0 my-project latest"
    exit 1
fi

PROJECT_NAME=$1
TAG=${2:-latest}
REGISTRY="gcr.io/mde-cloud/image-repo"
IMAGE_NAME="${REGISTRY}/${PROJECT_NAME}:python3.13-cu126-${TAG}"

echo -e "${YELLOW}========================================${NC}"
echo -e "${GREEN}Python 3.13 + CUDA 12.6 镜像构建${NC}"
echo -e "${YELLOW}========================================${NC}"
echo "项目名称: $PROJECT_NAME"
echo "标签: $TAG"
echo "镜像名称: $IMAGE_NAME"
echo -e "${YELLOW}========================================${NC}"
echo ""

# 询问是否继续
read -p "是否继续构建? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}构建已取消${NC}"
    exit 0
fi

# 构建镜像
echo -e "${GREEN}开始构建镜像...${NC}"
echo ""

docker build --platform linux/amd64 -t "$IMAGE_NAME" .

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ 镜像构建成功!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo "镜像名称: $IMAGE_NAME"
    echo ""
    
    # 显示镜像信息
    docker images "$IMAGE_NAME"
    echo ""
    
    # 询问是否推送
    read -p "是否推送镜像到 GCR? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}推送镜像到 GCR...${NC}"
        docker push "$IMAGE_NAME"
        
        if [ $? -eq 0 ]; then
            echo ""
            echo -e "${GREEN}✅ 镜像推送成功!${NC}"
            echo ""
            echo -e "${YELLOW}在 HPC 集群中使用:${NC}"
            echo "image: $IMAGE_NAME"
        else
            echo -e "${RED}❌ 镜像推送失败${NC}"
            exit 1
        fi
    else
        echo ""
        echo -e "${YELLOW}手动推送命令:${NC}"
        echo "docker push $IMAGE_NAME"
    fi
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}后续步骤:${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo "1. 在容器中安装 PyTorch:"
    echo "   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
    echo ""
    echo "2. 在容器中安装 vLLM:"
    echo "   VLLM_USE_PRECOMPILED=1 pip3 install vllm --extra-index-url https://download.pytorch.org/whl/cu124"
    echo ""
    echo "3. 测试镜像:"
    echo "   docker run --gpus all -it --rm $IMAGE_NAME /bin/bash"
    echo ""
    
else
    echo -e "${RED}❌ 镜像构建失败${NC}"
    exit 1
fi

