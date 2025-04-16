#!/bin/bash
# 如果任何命令返回错误则终止脚本
set -e

echo "开始克隆 lmcache-vllm 仓库..."
git clone https://github.com/LMCache/lmcache-vllm
cd lmcache-vllm
echo "安装 lmcache-vllm (editable 模式)..."
pip install -e .
cd ..

echo "开始克隆 LMCache 仓库..."
git clone https://github.com/LMCache/LMCache
cd LMCache
echo "安装 LMCache (editable 模式)..."
pip install -e .
cd ..

echo "安装 benchmarks/rag 目录下的依赖..."
pip install -r LMCache/benchmarks/rag/requirements.txt

echo "更新 apt 并安装 libibverbs 依赖..."
apt update
apt install -y libibverbs1 libibverbs-dev

echo "所有操作完成！"
