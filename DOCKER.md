# Radarm Docker 部署指南

本文档说明如何使用 Docker 部署 Radarm 项目。

## 📋 前置要求

- Docker Engine 20.10+
- Docker Compose 2.0+

## 🚀 快速开始

### 1. 生产环境部署

```bash
# 构建并启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

访问：
- 前端：http://localhost:3000
- 后端 API：http://localhost:8000
- API 文档：http://localhost:8000/docs

### 2. 开发环境部署（支持热重载）

```bash
# 使用开发配置启动
docker-compose -f docker-compose.dev.yml up

# 后台运行
docker-compose -f docker-compose.dev.yml up -d
```

开发模式下：
- 后端代码修改会自动重载（uvicorn --reload）
- 前端代码修改会自动刷新（React 热重载）

## 📁 项目结构

```
radarm/
├── backend/
│   └── Dockerfile          # 后端生产环境镜像
├── frontend/
│   ├── Dockerfile          # 前端生产环境镜像
│   ├── Dockerfile.dev      # 前端开发环境镜像
│   └── nginx.conf          # Nginx 配置
├── docker-compose.yml      # 生产环境配置
├── docker-compose.dev.yml  # 开发环境配置
└── .dockerignore           # Docker 构建忽略文件
```

## 🔧 配置说明

### 后端配置

- **基础镜像**：`python:3.10-slim`
- **端口**：8000
- **数据目录**：
  - `./radarm_data` → `/app/radarm_data`（会话数据）
  - `./out` → `/app/out`（图表输出）

### 前端配置

- **构建阶段**：`node:18-alpine`（构建 React 应用）
- **运行阶段**：`nginx:alpine`（静态文件服务）
- **端口**：3000（映射到容器内 80）

### 数据持久化

以下目录会挂载为 volumes，数据会持久化到宿主机：

- `radarm_data/` - 会话数据和元数据
- `out/` - 生成的图表和报告

## 🛠️ 常用命令

### 构建镜像

```bash
# 构建所有服务
docker-compose build

# 只构建后端
docker-compose build backend

# 只构建前端
docker-compose build frontend
```

### 查看日志

```bash
# 所有服务日志
docker-compose logs -f

# 只看后端日志
docker-compose logs -f backend

# 只看前端日志
docker-compose logs -f frontend
```

### 进入容器

```bash
# 进入后端容器
docker-compose exec backend bash

# 进入前端容器（生产环境是 nginx）
docker-compose exec frontend sh
```

### 重启服务

```bash
# 重启所有服务
docker-compose restart

# 重启特定服务
docker-compose restart backend
```

### 清理

```bash
# 停止并删除容器
docker-compose down

# 停止并删除容器、网络、volumes
docker-compose down -v

# 删除所有相关镜像
docker-compose down --rmi all
```

## 🔍 故障排查

### 后端无法启动

1. 检查端口是否被占用：
   ```bash
   netstat -ano | findstr :8000  # Windows
   lsof -i :8000                 # Linux/Mac
   ```

2. 查看后端日志：
   ```bash
   docker-compose logs backend
   ```

3. 检查依赖安装：
   ```bash
   docker-compose exec backend pip list
   ```

### 前端无法访问后端 API

1. 检查 nginx 配置是否正确代理了 API 请求
2. 确认前端代码中的 API_BASE 配置
3. 查看浏览器控制台的网络请求

### 中文字体显示问题

后端 Dockerfile 已安装 `fonts-wqy-zenhei` 和 `fonts-wqy-microhei`。如果图表中文仍显示为方框，可以：

1. 进入容器检查字体：
   ```bash
   docker-compose exec backend fc-list | grep -i wqy
   ```

2. 手动安装额外字体（如果需要）：
   ```bash
   docker-compose exec backend apt-get update
   docker-compose exec backend apt-get install -y fonts-noto-cjk
   ```

## 📝 环境变量

可以通过环境变量配置服务：

### 后端环境变量

- `PYTHONUNBUFFERED=1` - Python 输出不缓冲
- `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1` - 禁用 Fortran CTRL+C 处理

### 前端环境变量（开发模式）

- `REACT_APP_API_BASE` - API 基础地址（默认：http://localhost:8000）
- `CHOKIDAR_USEPOLLING=true` - 文件监听轮询（Docker 环境需要）

## 🚢 生产部署建议

1. **使用环境变量文件**：
   ```bash
   docker-compose --env-file .env.production up -d
   ```

2. **配置 HTTPS**：
   - 在 nginx 配置中添加 SSL 证书
   - 或使用反向代理（如 Traefik、Nginx Proxy Manager）

3. **资源限制**：
   在 `docker-compose.yml` 中添加：
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
   ```

4. **日志管理**：
   配置日志驱动和日志轮转

5. **健康检查**：
   已配置后端健康检查，可监控服务状态

## 📚 相关文档

- [Docker Compose 文档](https://docs.docker.com/compose/)
- [FastAPI 部署文档](https://fastapi.tiangolo.com/deployment/)
- [React 生产构建](https://create-react-app.dev/docs/production-build/)

