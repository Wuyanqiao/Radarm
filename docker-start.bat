@echo off
REM Radarm Docker 快速启动脚本 (Windows)

echo 🚀 Radarm Docker 启动脚本
echo ==========================
echo.

REM 检查 Docker 是否运行
echo 正在检查 Docker...
docker info >nul 2>&1
if errorlevel 1 (
    echo.
    echo ❌ 错误：Docker 未运行！
    echo.
    echo 请执行以下步骤：
    echo 1. 启动 Docker Desktop
    echo 2. 等待 Docker 完全启动（系统托盘图标不再闪烁）
    echo 3. 重新运行此脚本
    echo.
    pause
    exit /b 1
)
echo ✅ Docker 已运行

REM 检查 docker-compose 命令（支持 v1 和 v2）
echo 正在检查 Docker Compose...
docker-compose version >nul 2>&1
if errorlevel 1 (
    docker compose version >nul 2>&1
    if errorlevel 1 (
        echo.
        echo ❌ 错误：未找到 docker-compose 命令
        echo 请确保已安装 Docker Desktop（包含 Docker Compose）
        echo.
        pause
        exit /b 1
    ) else (
        set COMPOSE_CMD=docker compose
    )
) else (
    set COMPOSE_CMD=docker-compose
)
echo ✅ Docker Compose 可用

REM 选择模式
echo 请选择启动模式：
echo 1) 生产环境 (docker-compose.yml)
echo 2) 开发环境 (docker-compose.dev.yml)
set /p mode="请输入选项 [1/2] (默认: 1): "

if "%mode%"=="" set mode=1
if "%mode%"=="2" (
    set COMPOSE_FILE=docker-compose.dev.yml
    echo.
    echo 📦 使用开发环境配置...
) else (
    set COMPOSE_FILE=docker-compose.yml
    echo.
    echo 📦 使用生产环境配置...
)

REM 构建并启动
echo.
echo 🔨 构建镜像（这可能需要几分钟，请耐心等待）...
%COMPOSE_CMD% -f %COMPOSE_FILE% build
if errorlevel 1 (
    echo.
    echo ❌ 构建失败，请检查错误信息
    pause
    exit /b 1
)

echo.
echo 🚀 启动服务...
%COMPOSE_CMD% -f %COMPOSE_FILE% up -d
if errorlevel 1 (
    echo.
    echo ❌ 启动失败，请检查错误信息
    pause
    exit /b 1
)

echo.
echo ⏳ 等待服务启动（10秒）...
timeout /t 10 /nobreak >nul

REM 检查服务状态
echo.
echo 📊 服务状态：
%COMPOSE_CMD% -f %COMPOSE_FILE% ps

echo.
echo ✅ 启动完成！
echo.
echo 📍 访问地址：
echo   - 前端: http://localhost:3000
echo   - 后端 API: http://localhost:8000
echo   - API 文档: http://localhost:8000/docs
echo.
echo 💡 常用命令：
echo   查看日志: %COMPOSE_CMD% -f %COMPOSE_FILE% logs -f
echo   停止服务: %COMPOSE_CMD% -f %COMPOSE_FILE% down
echo   重启服务: %COMPOSE_CMD% -f %COMPOSE_FILE% restart

pause

