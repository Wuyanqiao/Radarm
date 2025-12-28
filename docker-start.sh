#!/bin/bash
# Radarm Docker 快速启动脚本

set -e

echo "🚀 Radarm Docker 启动脚本"
echo "=========================="
echo ""

# 检查 Docker 是否运行
if ! docker info > /dev/null 2>&1; then
    echo "❌ 错误：Docker 未运行，请先启动 Docker"
    exit 1
fi

# 检查 docker-compose 是否可用
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ 错误：未找到 docker-compose，请先安装 Docker Compose"
    exit 1
fi

# 选择模式
echo "请选择启动模式："
echo "1) 生产环境 (docker-compose.yml)"
echo "2) 开发环境 (docker-compose.dev.yml)"
read -p "请输入选项 [1/2] (默认: 1): " mode

mode=${mode:-1}

if [ "$mode" = "2" ]; then
    COMPOSE_FILE="docker-compose.dev.yml"
    echo ""
    echo "📦 使用开发环境配置..."
else
    COMPOSE_FILE="docker-compose.yml"
    echo ""
    echo "📦 使用生产环境配置..."
fi

# 构建并启动
echo ""
echo "🔨 构建镜像..."
docker-compose -f $COMPOSE_FILE build

echo ""
echo "🚀 启动服务..."
docker-compose -f $COMPOSE_FILE up -d

echo ""
echo "⏳ 等待服务启动..."
sleep 5

# 检查服务状态
echo ""
echo "📊 服务状态："
docker-compose -f $COMPOSE_FILE ps

echo ""
echo "✅ 启动完成！"
echo ""
echo "访问地址："
echo "  - 前端: http://localhost:3000"
echo "  - 后端 API: http://localhost:8000"
echo "  - API 文档: http://localhost:8000/docs"
echo ""
echo "查看日志: docker-compose -f $COMPOSE_FILE logs -f"
echo "停止服务: docker-compose -f $COMPOSE_FILE down"

