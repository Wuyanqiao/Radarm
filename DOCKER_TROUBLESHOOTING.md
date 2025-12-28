# Docker 故障排查指南

## 常见问题

### 1. ❌ Docker 未运行

**错误信息：**
```
❌ 错误：Docker 未运行，请先启动 Docker Desktop
```

**解决方案：**

1. **检查 Docker Desktop 是否安装**
   - Windows: 在开始菜单搜索 "Docker Desktop"
   - 如果未安装，下载：https://www.docker.com/products/docker-desktop

2. **启动 Docker Desktop**
   - 双击 Docker Desktop 图标
   - 等待 Docker 完全启动（系统托盘图标不再闪烁）
   - 确认状态栏显示 "Docker Desktop is running"

3. **验证 Docker 运行**
   ```bash
   docker --version
   docker info
   ```

4. **如果仍然无法连接**
   - 重启 Docker Desktop
   - 重启计算机
   - 检查 Windows 服务中 Docker 相关服务是否运行
   - 查看 Docker Desktop 的日志（Settings → Troubleshoot → View logs）

### 2. ❌ Docker Compose 未找到

**错误信息：**
```
❌ 错误：未找到 docker-compose 命令
```

**解决方案：**

1. **Docker Desktop 包含 Docker Compose**
   - 确保使用最新版本的 Docker Desktop
   - Docker Compose v2 使用 `docker compose`（注意空格）
   - Docker Compose v1 使用 `docker-compose`（注意连字符）

2. **检查版本**
   ```bash
   # 尝试 v2 命令
   docker compose version
   
   # 或尝试 v1 命令
   docker-compose version
   ```

3. **如果都不可用**
   - 重新安装 Docker Desktop
   - 确保安装时选择了 Docker Compose 组件

### 3. ❌ 端口被占用

**错误信息：**
```
Error: bind: address already in use
```

**解决方案：**

1. **检查端口占用**
   ```bash
   # Windows PowerShell
   netstat -ano | findstr :8000
   netstat -ano | findstr :3000
   ```

2. **修改端口映射**
   编辑 `docker-compose.yml`：
   ```yaml
   ports:
     - "8001:8000"  # 改为其他端口
     - "3001:80"
   ```

3. **停止占用端口的进程**
   ```bash
   # 找到进程 ID (PID) 后
   taskkill /PID <PID> /F
   ```

### 4. ❌ 构建失败

**可能原因：**

1. **网络问题**
   - 配置 Docker 镜像源（国内用户）
   - 在 Docker Desktop → Settings → Docker Engine 添加：
     ```json
     {
       "registry-mirrors": [
         "https://docker.mirrors.ustc.edu.cn",
         "https://hub-mirror.c.163.com"
       ]
     }
     ```

2. **依赖下载失败**
   - 检查网络连接
   - 尝试手动构建：
     ```bash
     docker-compose build --no-cache backend
     ```

3. **内存不足**
   - 增加 Docker Desktop 的内存分配
   - Settings → Resources → Advanced → Memory

### 5. ❌ 容器启动后立即退出

**排查步骤：**

1. **查看容器日志**
   ```bash
   docker-compose logs backend
   docker-compose logs frontend
   ```

2. **检查容器状态**
   ```bash
   docker-compose ps
   ```

3. **进入容器调试**
   ```bash
   docker-compose exec backend bash
   # 或
   docker run -it --entrypoint bash radarm-backend
   ```

### 6. ❌ 前端无法访问后端 API

**问题：**
前端页面可以打开，但无法连接到后端 API。

**解决方案：**

1. **检查 API 地址**
   - 前端代码中 `API_BASE` 配置
   - 在 Docker 环境中，前端通过 nginx 代理访问后端

2. **检查 nginx 配置**
   - 确认 `frontend/nginx.conf` 中的代理配置正确
   - 确认后端服务名称为 `backend`

3. **检查网络连接**
   ```bash
   # 在 frontend 容器中测试
   docker-compose exec frontend wget -O- http://backend:8000/docs
   ```

### 7. ❌ 中文字体显示为方框

**解决方案：**

后端 Dockerfile 已安装中文字体，如果仍有问题：

1. **检查字体是否安装**
   ```bash
   docker-compose exec backend fc-list | grep -i wqy
   ```

2. **手动安装字体**
   ```bash
   docker-compose exec backend apt-get update
   docker-compose exec backend apt-get install -y fonts-noto-cjk
   ```

3. **重启后端容器**
   ```bash
   docker-compose restart backend
   ```

## 诊断工具

### 运行环境检查脚本

**Windows:**
```bash
check-docker.bat
```

这个脚本会自动检查：
- Docker 命令是否可用
- Docker 服务是否运行
- Docker Compose 是否可用

## 获取帮助

如果以上方法都无法解决问题：

1. **收集信息**
   ```bash
   # Docker 版本
   docker --version
   docker compose version
   
   # Docker 信息
   docker info
   
   # 容器状态
   docker-compose ps
   
   # 容器日志
   docker-compose logs > docker-logs.txt
   ```

2. **查看详细日志**
   - Docker Desktop → Settings → Troubleshoot → View logs
   - 或运行：`docker-compose logs -f`

3. **重置 Docker**
   - Docker Desktop → Settings → Troubleshoot → Reset to factory defaults
   - ⚠️ 注意：这会删除所有容器和镜像

## 常用命令参考

```bash
# 查看所有容器
docker ps -a

# 查看容器日志
docker-compose logs -f [service_name]

# 重启服务
docker-compose restart [service_name]

# 停止所有服务
docker-compose down

# 停止并删除 volumes
docker-compose down -v

# 重新构建镜像
docker-compose build --no-cache

# 查看资源使用
docker stats

# 清理未使用的资源
docker system prune -a
```

