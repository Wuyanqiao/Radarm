#!/bin/bash

echo "===================================="
echo "  Radarm 前端服务启动脚本 (Linux/macOS)"
echo "===================================="
echo ""

# 检查 Node.js 是否安装
if ! command -v node &> /dev/null; then
    echo "[错误] 未检测到 Node.js，请先安装 Node.js v16 或更高版本"
    exit 1
fi

# 检查依赖是否安装
if [ ! -d "node_modules" ]; then
    echo "[信息] 检测到缺少依赖，正在安装..."
    npm install
    if [ $? -ne 0 ]; then
        echo "[错误] 依赖安装失败"
        exit 1
    fi
fi

echo "[信息] 正在启动前端开发服务器..."
echo "[信息] 服务地址: http://localhost:3000"
echo "[信息] 按 Ctrl+C 停止服务"
echo ""

npm start

