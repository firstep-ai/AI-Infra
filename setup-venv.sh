#!/bin/bash

# 虚拟环境设置脚本

echo "🔧 设置 Python 虚拟环境..."
echo ""

cd /Users/zhipeng.wang/Documents/GitHub/AI-Infra

# 1. 删除旧的虚拟环境（如果存在）
if [ -d ".venv" ]; then
    echo "📦 删除旧的虚拟环境..."
    rm -rf .venv
fi

# 2. 使用系统 Python 创建新虚拟环境
echo "🆕 创建新的虚拟环境..."
/usr/bin/python3 -m venv .venv

# 3. 升级 pip
echo "⬆️  升级 pip..."
.venv/bin/pip install --upgrade pip

# 4. 安装所需包
echo "📥 安装所需的 Python 包..."
.venv/bin/pip install ipykernel python-dotenv openai

echo ""
echo "✅ 虚拟环境设置完成！"
echo ""
echo "📝 使用方法："
echo "   1. 激活虚拟环境: source .venv/bin/activate"
echo "   2. 运行脚本: python openai_test.py"
echo "   3. 退出虚拟环境: deactivate"
echo ""
echo "💡 Jupyter Notebook 使用："
echo "   在 VSCode/Cursor 中，选择 .venv/bin/python 作为 kernel"
echo ""


