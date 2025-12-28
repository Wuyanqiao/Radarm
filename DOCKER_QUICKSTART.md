# 🐳 Radarm Docker 快速启动指南

## 一键启动（推荐）

### Windows
```bash
docker-start.bat
```

### Linux/Mac
```bash
chmod +x docker-start.sh
./docker-start.sh
```

## 手动启动

### 生产环境
```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止
docker-compose down
```

### 开发环境（支持热重载）
```bash
# 启动开发环境
docker-compose -f docker-compose.dev.yml up

# 后台运行
docker-compose -f docker-compose.dev.yml up -d
```

## 访问地址

启动成功后访问：
- 🌐 **前端**: http://localhost:3000
- 🔌 **后端 API**: http://localhost:8000  
- 📚 **API 文档**: http://localhost:8000/docs

## 常用命令

```bash
# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f backend    # 后端日志
docker-compose logs -f frontend   # 前端日志

# 重启服务
docker-compose restart backend

# 进入容器
docker-compose exec backend bash

# 停止并删除容器
docker-compose down

# 完全清理（包括 volumes）
docker-compose down -v
```

## 数据持久化

以下目录会自动持久化到宿主机：
- `./radarm_data` - 会话数据
- `./out` - 生成的图表和报告

## 故障排查

1. **端口被占用**：修改 `docker-compose.yml` 中的端口映射
2. **构建失败**：检查网络连接，可能需要配置镜像源
3. **中文显示问题**：后端已安装中文字体，如仍有问题请查看日志

更多详细信息请参考 [DOCKER.md](./DOCKER.md)

