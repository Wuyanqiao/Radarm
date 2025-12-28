@echo off
chcp 65001 >nul
echo ====================================
echo   Radarm 前端服务启动脚本 (Windows)
echo ====================================
echo.

REM 检查 Node.js 是否安装
node --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Node.js，请先安装 Node.js v16 或更高版本
    pause
    exit /b 1
)

REM 检查依赖是否安装
if not exist "node_modules" (
    echo [信息] 检测到缺少依赖，正在安装...
    call npm install
    if errorlevel 1 (
        echo [错误] 依赖安装失败
        pause
        exit /b 1
    )
)

echo [信息] 正在启动前端开发服务器...
echo [信息] 服务地址: http://localhost:3000
echo [信息] 按 Ctrl+C 停止服务
echo.

call npm start

pause

