# Radarm 快速开始指南 ⚡

这是 Radarm 的最简快速开始指南。如果你需要更详细的说明，请查看 [DEPLOYMENT.md](./DEPLOYMENT.md)。

## 🚀 三种启动方式（选择其一）

### 方式一：使用启动脚本（最简单）✨

**Windows:**
```bash
# 终端1：启动后端
start.bat

# 终端2：启动前端
start-frontend.bat
```

**Linux/macOS:**
```bash
# 首次使用需要添加执行权限
chmod +x start.sh start-frontend.sh

# 终端1：启动后端
./start.sh

# 终端2：启动前端
./start-frontend.sh
```

### 方式二：Docker 一键启动 🐳

```bash
# 一键启动（包含前端和后端）
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

访问：http://localhost:8000

### 方式三：手动启动 🔧

#### 1. 安装依赖

**前端：**
```bash
npm install
```

**后端：**
```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

#### 2. 启动服务

**终端1 - 后端：**
```bash
python backend.py
```

**终端2 - 前端：**
```bash
npm start
```

## 📝 首次使用

1. 打开浏览器访问 http://localhost:3000（本地部署）或 http://localhost:8000（Docker部署）

2. 点击左下角的 **设置图标**，配置 API Keys：
   - DeepSeek API Key（用于 DeepSeek-A/B/C）
   - 智谱AI API Key（用于 GLM 模型和视觉模型）
   - 通义千问 API Key（用于 Qwen 模型和视觉模型）

3. 点击顶部的 **"导入文件"** 或 **"连接数据库"** 导入数据

4. 开始使用！

## ❓ 遇到问题？

- 查看 [DEPLOYMENT.md](./DEPLOYMENT.md) 的常见问题部分
- 检查端口是否被占用（前端3000，后端8000）
- 确保已安装所有依赖

## 📚 更多资源

- [完整部署文档](./DEPLOYMENT.md) - 详细的部署说明
- [环境变量配置](./env.example) - 环境变量示例
- [README.md](./README.md) - 功能使用说明

