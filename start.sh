#!/bin/bash

# 昇腾RAG系统启动脚本

echo "启动昇腾RAG系统..."

# 激活虚拟环境
source venv/bin/activate

# 检查模型是否存在
if [ ! -d "models/embeddings/sentence-transformers" ]; then
    echo "警告: 嵌入模型未找到，请先运行 bash models/download_models.sh"
fi

# 设置环境变量
export PYTHONPATH=$PWD/src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# 启动API服务器（后台运行）
echo "启动API服务器..."
nohup python src/api_server.py > logs/api_server.log 2>&1 &
API_PID=$!
echo "API服务器PID: $API_PID"

# 等待API服务器启动
sleep 5

# 启动Streamlit Web界面
echo "启动Web界面..."
streamlit run web/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# 清理函数
cleanup() {
    echo "正在关闭服务..."
    kill $API_PID 2>/dev/null
    exit 0
}

# 捕获中断信号
trap cleanup SIGINT SIGTERM

# 等待
wait