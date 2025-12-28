# Radarm 部署指南 🚀

本文档提供 Radarm 项目的详细部署说明，包括本地开发环境、Docker 容器化部署、生产环境部署等多种方式。

## 📋 目录

- [系统要求](#系统要求)
- [部署方式](#部署方式)
  - [方式一：本地开发环境部署](#方式一本地开发环境部署)
  - [方式二：Docker 容器化部署](#方式二docker-容器化部署)
  - [方式三：生产环境部署](#方式三生产环境部署)
- [配置说明](#配置说明)
- [常见问题](#常见问题)
- [性能优化](#性能优化)

---

## 系统要求

### 最低配置
- **操作系统**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **CPU**: 2 核心
- **内存**: 4GB RAM
- **磁盘空间**: 5GB 可用空间

### 推荐配置
- **CPU**: 4+ 核心
- **内存**: 8GB+ RAM
- **磁盘空间**: 20GB+ 可用空间（用于数据存储）

### 必需软件

#### 方式一：本地部署
- **Node.js**: v16.0.0 或更高版本
- **Python**: 3.8 或更高版本（推荐 3.11+）
- **pip**: Python 包管理器
- **npm**: Node.js 包管理器（随 Node.js 安装）

#### 方式二：Docker 部署
- **Docker**: 20.10 或更高版本
- **Docker Compose**: 1.29 或更高版本（可选，用于一键部署）

---

## 部署方式

### 方式一：本地开发环境部署

这是最简单的部署方式，适合开发和测试使用。

#### 1. 克隆项目

```bash
git clone <repository-url>
cd radarm
```

#### 2. 配置环境变量（可选）

```bash
# 复制环境变量示例文件
cp env.example .env

# 编辑 .env 文件，填入你的 API Keys
# 注意：API Keys 也可以在前端界面中配置，无需在此设置
```

#### 3. 安装前端依赖

```bash
# 安装 Node.js 依赖
npm install

# 如果遇到依赖问题，尝试清理缓存
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

#### 4. 安装后端依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 安装 Python 依赖
pip install -r requirements.txt

# 如果下载慢，使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 5. 启动服务

**启动后端（第一个终端窗口）:**

```bash
python backend.py
```

后端将在 `http://0.0.0.0:8000` 启动。

**启动前端（第二个终端窗口）:**

```bash
npm start
```

前端将在 `http://localhost:3000` 启动，浏览器会自动打开。

#### 6. 验证部署

- 打开浏览器访问 `http://localhost:3000`
- 点击设置图标，配置 API Keys
- 尝试上传一个 CSV 文件测试功能

---

### 方式二：Docker 容器化部署

适合需要隔离环境、一键部署的场景。

#### 1. 准备工作

确保已安装 Docker 和 Docker Compose：

```bash
# 检查 Docker 版本
docker --version
docker-compose --version
```

#### 2. 配置环境变量

```bash
# 复制环境变量示例文件
cp env.example .env

# 根据需要编辑 .env 文件（可选）
```

#### 3. 使用 Docker Compose 部署（推荐）

```bash
# 构建并启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down

# 停止服务并删除数据卷
docker-compose down -v
```

#### 4. 使用 Docker 单独构建

```bash
# 构建镜像
docker build -t radarm:latest .

# 运行容器
docker run -d \
  --name radarm-app \
  -p 8000:8000 \
  -v $(pwd)/radarm_data:/app/radarm_data \
  -v $(pwd)/out:/app/out \
  --env-file .env \
  radarm:latest

# 查看日志
docker logs -f radarm-app

# 停止容器
docker stop radarm-app
docker rm radarm-app
```

#### 5. 访问应用

- 打开浏览器访问 `http://localhost:8000`
- Docker 部署会直接使用构建好的前端，无需单独启动前端服务

---

### 方式三：生产环境部署

#### 使用 Nginx 反向代理（推荐）

##### 1. 构建生产版本前端

```bash
cd radarm
npm run build
```

##### 2. 配置 Nginx

创建 `/etc/nginx/sites-available/radarm` 配置文件：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # 前端静态文件
    location / {
        root /path/to/radarm/build;
        try_files $uri $uri/ /index.html;
    }

    # 后端 API 代理
    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket 支持（如果需要）
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

##### 3. 启用 Nginx 配置

```bash
# 创建软链接
sudo ln -s /etc/nginx/sites-available/radarm /etc/nginx/sites-enabled/

# 测试配置
sudo nginx -t

# 重启 Nginx
sudo systemctl restart nginx
```

##### 4. 使用 systemd 管理后端服务

创建 `/etc/systemd/system/radarm.service`：

```ini
[Unit]
Description=Radarm Backend Service
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/radarm
Environment="PATH=/path/to/radarm/venv/bin"
ExecStart=/path/to/radarm/venv/bin/python backend.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable radarm
sudo systemctl start radarm
sudo systemctl status radarm
```

#### 使用 Gunicorn + Uvicorn Workers（可选）

如果需要更好的生产性能，可以使用 Gunicorn：

```bash
pip install gunicorn

# 修改启动方式
gunicorn backend:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120
```

---

## 配置说明

### 环境变量

详细的环境变量说明请参考 `env.example` 文件。主要配置项：

#### API 配置
- `DEEPSEEK_BASE_URL`: DeepSeek API 基础 URL
- `ZHIPU_BASE_URL`: 智谱AI API 基础 URL
- `QWEN_BASE_URL`: 通义千问 API 基础 URL

#### 视觉模型配置
- `VISION_MAX_IMAGES`: 最大图片数量（默认 3）
- `VISION_MAX_EDGE`: 图片最大边长（默认 1024）
- `VISION_TIMEOUT`: 视觉模型请求超时时间（默认 120 秒）

### 数据存储

- **会话数据**: `radarm_data/sessions/` - 存储用户会话和原始数据
- **输出文件**: `out/` - 存储生成的图表和报告

### 端口配置

- **前端开发服务器**: 3000（仅开发模式）
- **后端 API 服务器**: 8000
- **生产环境**: 通常通过 Nginx 在 80/443 端口提供服务

---

## 常见问题

### Q1: 前端启动失败，提示端口被占用

**解决方案:**
```bash
# Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:3000 | xargs kill -9
```

或修改端口：
```bash
# 在 package.json 中修改
"start": "PORT=3001 react-scripts start"
```

### Q2: 后端启动失败，提示模块未找到

**解决方案:**
```bash
# 确保在虚拟环境中
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# 重新安装依赖
pip install -r requirements.txt
```

### Q3: 图表中文显示为方框

**解决方案（Linux）:**
```bash
# Ubuntu/Debian
sudo apt-get install fonts-wqy-zenhei

# CentOS/RHEL
sudo yum install wqy-zenhei-fonts
```

### Q4: Docker 构建失败

**解决方案:**
- 检查 Dockerfile 中的路径是否正确
- 确保有足够的磁盘空间
- 尝试清理 Docker 缓存：`docker system prune -a`

### Q5: 上传大文件时超时

**解决方案:**
- 增加后端超时时间
- 使用 Nginx 配置增加 `client_max_body_size`
- 考虑使用分块上传

---

## 性能优化

### 前端优化

1. **启用生产构建**:
```bash
npm run build
# 使用构建后的 build 目录部署
```

2. **启用 Gzip 压缩**（Nginx）:
```nginx
gzip on;
gzip_types text/plain text/css application/json application/javascript;
```

### 后端优化

1. **使用多进程部署**:
```bash
# 使用 Gunicorn + Uvicorn workers
gunicorn backend:app -w 4 -k uvicorn.workers.UvicornWorker
```

2. **启用缓存**:
   - 考虑使用 Redis 缓存会话数据
   - 缓存常用的分析结果

3. **数据库连接池**:
   - 如果使用数据库连接，配置连接池大小

### 系统优化

1. **限制资源使用**:
```yaml
# docker-compose.yml 中添加
services:
  radarm:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

2. **定期清理**:
   - 定期清理 `out/` 目录中的旧文件
   - 清理不再使用的会话数据

---

## 安全建议

1. **API Keys 安全**:
   - 不要在代码中硬编码 API Keys
   - 使用环境变量或密钥管理服务
   - `.env` 文件不要提交到版本控制

2. **生产环境**:
   - 使用 HTTPS
   - 配置防火墙规则
   - 定期更新依赖包
   - 使用非 root 用户运行服务

3. **数据备份**:
   - 定期备份 `radarm_data/` 目录
   - 备份重要的报告和分析结果

---

## 更新升级

### 更新代码

```bash
# 拉取最新代码
git pull origin main

# 更新前端依赖
npm install

# 更新后端依赖
pip install -r requirements.txt --upgrade

# 重启服务
```

### Docker 更新

```bash
# 停止服务
docker-compose down

# 重新构建
docker-compose build --no-cache

# 启动服务
docker-compose up -d
```

---

## 技术支持

如果遇到问题，请：

1. 查看本文档的 [常见问题](#常见问题) 部分
2. 检查日志文件
3. 提交 Issue 到项目仓库

---

**最后更新**: 2024年
**文档版本**: 1.0

