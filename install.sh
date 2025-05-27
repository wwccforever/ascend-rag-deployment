#!/bin/bash

# 昇腾RAG系统安装脚本

echo "开始安装昇腾RAG系统..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "Python版本检查通过: $python_version"
else
    echo "错误: 需要Python 3.8或更高版本，当前版本: $python_version"
    exit 1
fi

# 创建虚拟环境
echo "创建虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装依赖
echo "安装Python依赖包..."
pip install -r requirements.txt

# 创建必要的目录
echo "创建目录结构..."
mkdir -p data/documents
mkdir -p data/vector_store
mkdir -p logs
mkdir -p models/embeddings
mkdir -p models/llm

# 设置权限
chmod +x start.sh
chmod +x models/download_models.sh

echo "安装完成！"
echo "请运行以下命令下载模型:"
echo "bash models/download_models.sh"
echo "然后运行以下命令启动服务:"
echo "bash start.sh"