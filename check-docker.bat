@echo off
REM Docker 环境检查脚本

echo ========================================
echo   Radarm Docker 环境检查
echo ========================================
echo.

REM 检查 Docker 命令是否存在
where docker >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker 命令未找到
    echo.
    echo 可能的原因：
    echo 1. Docker Desktop 未安装
    echo 2. Docker Desktop 未添加到 PATH 环境变量
    echo.
    echo 解决方案：
    echo 1. 下载并安装 Docker Desktop: https://www.docker.com/products/docker-desktop
    echo 2. 安装后重启计算机
    echo 3. 启动 Docker Desktop 并等待完全启动
    echo.
    pause
    exit /b 1
) else (
    echo ✅ Docker 命令已找到
    docker --version
)

echo.

REM 检查 Docker 是否运行
echo 正在检查 Docker 服务状态...
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker 服务未运行
    echo.
    echo 请执行以下步骤：
    echo 1. 打开 Docker Desktop 应用程序
    echo 2. 等待 Docker 完全启动（系统托盘图标不再闪烁）
    echo 3. 确认 Docker Desktop 显示 "Docker Desktop is running"
    echo.
    echo 如果 Docker Desktop 已启动但仍无法连接，请尝试：
    echo - 重启 Docker Desktop
    echo - 检查防火墙设置
    echo - 查看 Docker Desktop 的日志
    echo.
) else (
    echo ✅ Docker 服务正在运行
    echo.
    echo Docker 信息：
    docker info | findstr /C:"Server Version" /C:"Operating System" /C:"Total Memory"
)

echo.

REM 检查 Docker Compose
echo 正在检查 Docker Compose...
docker-compose version >nul 2>&1
if errorlevel 1 (
    docker compose version >nul 2>&1
    if errorlevel 1 (
        echo ❌ Docker Compose 未找到
        echo.
        echo Docker Desktop 应该包含 Docker Compose
        echo 如果未找到，请重新安装 Docker Desktop
        echo.
    ) else (
        echo ✅ Docker Compose v2 已找到
        docker compose version
    )
) else (
    echo ✅ Docker Compose v1 已找到
    docker-compose version
)

echo.
echo ========================================
echo   检查完成
echo ========================================
echo.

pause

