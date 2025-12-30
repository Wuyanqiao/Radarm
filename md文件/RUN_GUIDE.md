# Radarm 项目运行指南 🚀

本文档提供 Radarm 项目的完整运行指南，包括环境准备、配置、启动、验证等详细步骤。

## 📋 目录

- [系统要求](#系统要求)
- [快速开始（Docker Compose 推荐）](#快速开始docker-compose-推荐)
- [详细部署步骤](#详细部署步骤)
- [配置说明](#配置说明)
- [验证服务](#验证服务)
- [常见问题排查](#常见问题排查)
- [生产环境部署建议](#生产环境部署建议)

---

## 系统要求

### 最低配置
- **操作系统**: 
  - Windows 10+ (WSL2 推荐)
  - macOS 10.14+
  - Linux (Ubuntu 18.04+ / CentOS 7+)
- **CPU**: 2 核心
- **内存**: 4GB RAM（推荐 8GB+）
- **磁盘空间**: 10GB+ 可用空间

### 必需软件

#### Docker 部署（推荐）
- **Docker**: 20.10+ 
- **Docker Compose**: 2.0+（随 Docker Desktop 安装）

#### 本地开发部署（可选）
- **Node.js**: v18.0.0+
- **Python**: 3.11+
- **pip**: Python 包管理器
- **npm**: Node.js 包管理器

---

## 快速开始（Docker Compose 推荐）

### 1. 克隆项目

```bash
git clone <repository-url>
cd radarm
```

### 2. 配置环境变量（可选）

```bash
# 复制环境变量示例文件
cp env.example .env

# 编辑 .env 文件（可选，API Keys 也可在前端界面配置）
# Windows: notepad .env
# Linux/macOS: nano .env
```

**注意**：`.env` 文件主要用于 API 配置，如果在前端界面配置 API Keys，可以跳过此步骤。

### 3. 启动所有服务

```bash
# 构建并启动所有容器
docker compose up --build

# 或后台运行
docker compose up -d --build
```

### 4. 访问应用

- **前端界面**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health
- **Dask Dashboard**: http://localhost:8787/status

### 5. 停止服务

```bash
# 停止所有容器
docker compose down

# 停止并删除数据卷（谨慎使用）
docker compose down -v
```

---

## 详细部署步骤

### 方式一：Docker Compose 部署（生产推荐）

#### 步骤 1: 环境准备

确保已安装 Docker 和 Docker Compose：

```bash
# 检查 Docker 版本
docker --version
# 应显示: Docker version 20.10.x 或更高

# 检查 Docker Compose 版本
docker compose version
# 应显示: Docker Compose version v2.x.x 或更高
```

#### 步骤 2: 配置环境变量

创建 `.env` 文件（可选）：

```bash
cp env.example .env
```

编辑 `.env` 文件，配置以下内容（可选）：

```env
# API 配置（可选，也可在前端界面配置）
DEEPSEEK_BASE_URL=https://api.deepseek.com/chat/completions
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4/chat/completions
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions

# 数据库配置（可选，使用默认值也可）
POSTGRES_USER=radarm
POSTGRES_PASSWORD=radarm123
POSTGRES_DB=radarm
```

**重要提示**：
- API Keys 可以在前端界面中配置，无需在此设置
- 数据库密码建议在生产环境中修改为强密码

#### 步骤 3: 构建和启动

```bash
# 首次启动：构建镜像并启动所有服务
docker compose up --build -d

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f

# 查看特定服务日志
docker compose logs -f radarm
docker compose logs -f celery-worker
```

#### 步骤 4: 验证服务

等待所有服务启动完成（约 30-60 秒），然后验证：

```bash
# 检查健康状态
curl http://localhost:8000/health

# 应该返回: {"status":"ok","service":"radarm"}
```

#### 步骤 5: 访问应用

打开浏览器访问：http://localhost:8000

---

### 方式二：本地开发部署

适合需要修改代码的开发场景。

#### 步骤 1: 安装前端依赖

```bash
# 安装 Node.js 依赖
npm install

# 如果遇到依赖问题，清理缓存后重试
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

#### 步骤 2: 安装后端依赖

```bash
# 创建 Python 虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 安装 Python 依赖
pip install -r requirements.txt

# 如果下载慢，使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 步骤 3: 启动 Redis 和 PostgreSQL（必需）

**选项 A: 使用 Docker 启动依赖服务**

```bash
# 只启动 Redis 和 PostgreSQL
docker compose up -d redis postgres
```

**选项 B: 本地安装**

- **Redis**: 安装并启动 Redis 服务（默认端口 6379）
- **PostgreSQL**: 安装并创建数据库（默认端口 5432）

#### 步骤 4: 配置环境变量

创建 `.env` 文件并配置：

```env
BROKER_URL=redis://localhost:6379/0
RESULT_BACKEND=redis://localhost:6379/0
DATABASE_URL=postgresql+psycopg2://radarm:radarm123@localhost:5432/radarm
```

#### 步骤 5: 启动后端服务

```bash
# 启动 FastAPI 后端
python backend.py

# 或使用 uvicorn
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```

#### 步骤 6: 启动 Celery Worker（新终端）

```bash
# 激活虚拟环境
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# 启动 Celery Worker
celery -A tasks worker -l info --concurrency=2
```

#### 步骤 7: 启动前端（新终端，可选）

```bash
# 开发模式启动前端
npm start

# 或构建静态文件（后端已包含构建产物）
npm run build
```

#### 步骤 8: 访问应用

- **前端**: http://localhost:3000（开发模式）或 http://localhost:8000（生产模式）
- **后端 API**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs

---

## 配置说明

### 服务端口

| 服务 | 端口 | 说明 |
|------|------|------|
| Radarm App | 8000 | 主应用和 API |
| Redis | 6379 | 任务队列和缓存 |
| PostgreSQL | 5432 | 数据库 |
| Dask Scheduler | 8786 | Dask 调度器 |
| Dask Dashboard | 8787 | Dask 监控面板 |

### 数据目录

项目会在当前目录创建以下数据目录：

- `radarm_data/`: 会话数据、上传文件、工作区
  - `sessions/`: 会话存储
  - `uploads/`: 用户上传的文件
  - `workspaces/`: 沙箱工作区
- `out/`: 生成的图表和报告
  - `{session_id}/`: 每个会话的输出
  - `{session_id}/reports/`: 报告文件

### 环境变量说明

详见 `env.example` 文件，主要配置项：

- **API 配置**: DeepSeek、智谱、通义千问的 API 地址和模型
- **数据库配置**: PostgreSQL 连接信息
- **任务队列**: Redis 连接信息（Docker 部署自动配置）

---

## 验证服务

### 1. 健康检查

```bash
# 检查主应用
curl http://localhost:8000/health
# 预期: {"status":"ok","service":"radarm"}

# 检查 Redis
docker compose exec redis redis-cli ping
# 预期: PONG

# 检查 PostgreSQL
docker compose exec postgres pg_isready -U radarm
# 预期: postgres:5432 - accepting connections
```

### 2. 检查服务状态

```bash
# 查看所有容器状态
docker compose ps

# 应该看到以下服务都在运行：
# - radarm-app (healthy)
# - radarm-redis (healthy)
# - radarm-postgres (healthy)
# - radarm-celery-worker
# - radarm-dask-scheduler
# - radarm-dask-worker
```

### 3. 查看日志

```bash
# 查看所有服务日志
docker compose logs -f

# 查看特定服务日志
docker compose logs -f radarm
docker compose logs -f celery-worker
docker compose logs -f postgres
```

### 4. 测试 API

```bash
# 测试健康检查端点
curl http://localhost:8000/health

# 测试任务查询（需要先有任务）
curl http://localhost:8000/tasks/{task_id}
```

### 5. 前端功能测试

1. 打开 http://localhost:8000
2. 上传一个 CSV/Excel 文件
3. 尝试对话分析功能
4. 检查是否能正常生成图表和报告

---

## 常见问题排查

### 问题 1: 容器启动失败

**症状**: `docker compose up` 后容器立即退出

**排查步骤**:

```bash
# 查看详细错误日志
docker compose logs radarm

# 检查端口占用
# Windows: netstat -ano | findstr :8000
# Linux/macOS: lsof -i :8000

# 检查 Docker 资源
docker system df
docker system prune  # 清理未使用的资源
```

**常见原因**:
- 端口被占用：修改 `docker-compose.yml` 中的端口映射
- 内存不足：增加 Docker 内存限制
- 文件权限问题：检查 `radarm_data` 和 `out` 目录权限

### 问题 2: 沙箱功能不可用

**症状**: 代码执行时提示 "Docker 不可用，沙箱功能被禁用"

**排查步骤**:

```bash
# 检查 Docker Socket 挂载
docker compose exec radarm ls -la /var/run/docker.sock

# 检查容器内 Docker 客户端
docker compose exec radarm docker ps
```

**解决方案**:
- 确保 `docker-compose.yml` 中挂载了 `/var/run/docker.sock`
- 如果不需要沙箱功能，代码会自动回退到本地执行模式

### 问题 3: Celery 任务不执行

**症状**: 提交异步任务后一直处于 PENDING 状态

**排查步骤**:

```bash
# 检查 Celery Worker 是否运行
docker compose ps celery-worker

# 查看 Celery Worker 日志
docker compose logs -f celery-worker

# 检查 Redis 连接
docker compose exec redis redis-cli ping
```

**解决方案**:
- 确保 Redis 服务正常运行
- 检查 `BROKER_URL` 和 `RESULT_BACKEND` 环境变量
- 重启 Celery Worker: `docker compose restart celery-worker`

### 问题 4: 数据库连接失败

**症状**: 应用启动时报数据库连接错误

**排查步骤**:

```bash
# 检查 PostgreSQL 是否运行
docker compose ps postgres

# 检查数据库连接
docker compose exec postgres psql -U radarm -d radarm -c "SELECT 1;"

# 查看数据库日志
docker compose logs postgres
```

**解决方案**:
- 确保 PostgreSQL 容器正常运行
- 检查 `DATABASE_URL` 环境变量
- 等待数据库初始化完成（首次启动需要 10-30 秒）

### 问题 5: Dask Worker 启动失败

**症状**: Dask Worker 容器不断重启

**排查步骤**:

```bash
# 查看 Dask Worker 日志
docker compose logs dask-worker

# 检查 Dask Scheduler 是否运行
docker compose ps dask-scheduler
```

**解决方案**:
- 确保 Dask Scheduler 先启动
- 检查网络连接：Worker 需要能连接到 Scheduler

### 问题 6: 前端无法访问

**症状**: 浏览器访问 http://localhost:8000 显示错误

**排查步骤**:

```bash
# 检查后端服务是否运行
curl http://localhost:8000/health

# 检查前端构建产物
docker compose exec radarm ls -la /app/build

# 查看后端日志
docker compose logs -f radarm
```

**解决方案**:
- 确保后端服务正常运行
- 检查防火墙设置
- 清除浏览器缓存

### 问题 7: 内存不足

**症状**: 容器频繁 OOM（Out of Memory）

**解决方案**:

```bash
# 增加 Docker 内存限制（Docker Desktop）
# Settings -> Resources -> Memory -> 调整为 8GB+

# 或减少并发数
# 编辑 docker-compose.yml，修改 celery-worker 的 --concurrency=1
```

---

## 生产环境部署建议

### 1. 安全配置

#### 修改默认密码

```bash
# 编辑 .env 文件，设置强密码
POSTGRES_PASSWORD=your_strong_password_here
```

#### 配置 HTTPS

使用 Nginx 反向代理并配置 SSL 证书：

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### 限制 Docker Socket 访问

如果不需要沙箱功能，移除 `docker-compose.yml` 中的 Docker Socket 挂载：

```yaml
# 注释或删除这一行
# - /var/run/docker.sock:/var/run/docker.sock
```

### 2. 性能优化

#### 调整资源限制

编辑 `docker-compose.yml`，添加资源限制：

```yaml
services:
  radarm:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

#### 数据库优化

```sql
-- 连接 PostgreSQL
docker compose exec postgres psql -U radarm -d radarm

-- 创建索引（根据实际使用情况）
CREATE INDEX idx_sessions_created ON sessions(created_at);
```

#### Redis 持久化

已配置 AOF 持久化，确保数据不丢失。

### 3. 监控和日志

#### 配置日志轮转

```yaml
services:
  radarm:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

#### 使用监控工具

- **Prometheus + Grafana**: 监控服务指标
- **ELK Stack**: 集中日志管理
- **Dask Dashboard**: http://localhost:8787/status

### 4. 备份策略

#### 数据库备份

```bash
# 定期备份 PostgreSQL
docker compose exec postgres pg_dump -U radarm radarm > backup_$(date +%Y%m%d).sql

# 恢复备份
docker compose exec -T postgres psql -U radarm radarm < backup_20231230.sql
```

#### 数据目录备份

```bash
# 备份会话数据和输出
tar -czf radarm_backup_$(date +%Y%m%d).tar.gz radarm_data/ out/
```

### 5. 高可用部署

#### 多实例部署

```yaml
# 扩展 Celery Worker
docker compose up -d --scale celery-worker=3

# 使用负载均衡器（Nginx/HAProxy）分发请求
```

#### 外部数据库

使用云数据库（RDS/Azure Database）替代容器内数据库：

```yaml
# 修改 DATABASE_URL 环境变量
DATABASE_URL=postgresql+psycopg2://user:pass@external-db:5432/radarm
```

---

## 维护命令

### 日常维护

```bash
# 查看服务状态
docker compose ps

# 查看资源使用
docker stats

# 清理未使用的镜像和容器
docker system prune -a

# 更新代码后重新构建
docker compose up -d --build

# 重启特定服务
docker compose restart radarm
docker compose restart celery-worker
```

### 数据管理

```bash
# 查看数据目录大小
du -sh radarm_data/ out/

# 清理旧会话数据（谨慎操作）
find radarm_data/sessions -mtime +30 -type d -exec rm -rf {} \;

# 清理旧输出文件
find out/ -mtime +7 -type f -delete
```

---

## 获取帮助

如果遇到问题：

1. 查看日志：`docker compose logs -f`
2. 检查健康状态：`curl http://localhost:8000/health`
3. 查看本文档的"常见问题排查"部分
4. 提交 Issue 到项目仓库

---

## 更新日志

- **2025-12-30**: 初始版本，包含 Docker Compose 部署、Celery 异步任务、沙箱功能

---

**祝使用愉快！** 🎉

