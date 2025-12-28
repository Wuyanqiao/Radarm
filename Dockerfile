# Radarm 多阶段构建 Dockerfile
# 阶段1: 构建前端
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# 复制前端依赖文件
COPY package*.json ./
COPY tailwind.config.js ./
COPY postcss.config.js ./

# 安装前端依赖
RUN npm ci --only=production=false

# 复制前端源代码
COPY public ./public
COPY src ./src

# 构建前端
RUN npm run build

# 阶段2: Python 后端运行时
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（用于matplotlib和其他库）
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 复制Python依赖文件
COPY requirements.txt .

# 安装Python依赖（使用国内镜像加速，可选）
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple || \
    pip install --no-cache-dir -r requirements.txt

# 复制后端源代码
COPY *.py ./

# 从构建阶段复制前端构建产物
COPY --from=frontend-builder /app/frontend/build ./build

# 创建必要的目录
RUN mkdir -p radarm_data/sessions out

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV FOR_DISABLE_CONSOLE_CTRL_HANDLER=1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "backend.py"]

