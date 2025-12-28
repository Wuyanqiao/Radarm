#!/bin/bash

echo "===================================="
echo "  Radarm 后端服务启动脚本 (Linux/macOS)"
echo "===================================="
echo ""

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到 Python3，请先安装 Python 3.8 或更高版本"
    exit 1
fi

# 检查是否存在虚拟环境
if [ -f "venv/bin/activate" ]; then
    echo "[信息] 检测到虚拟环境，正在激活..."
    source venv/bin/activate
else
    echo "[警告] 未检测到虚拟环境，使用系统 Python"
    echo "[提示] 建议创建虚拟环境：python3 -m venv venv"
fi

# 检查依赖是否安装
python3 -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[信息] 检测到缺少依赖，正在安装..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[错误] 依赖安装失败"
        exit 1
    fi
fi

# 创建必要的目录
mkdir -p radarm_data/sessions
mkdir -p out

echo "[信息] 正在启动后端服务..."
echo "[信息] 服务地址: http://localhost:8000"
echo "[信息] 按 Ctrl+C 停止服务"
echo ""

python3 backend.py

