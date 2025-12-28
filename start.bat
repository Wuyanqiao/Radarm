@echo off
chcp 65001 >nul
echo ====================================
echo   Radarm 后端服务启动脚本 (Windows)
echo ====================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.8 或更高版本
    pause
    exit /b 1
)

REM 检查是否存在虚拟环境
if exist "venv\Scripts\activate.bat" (
    echo [信息] 检测到虚拟环境，正在激活...
    call venv\Scripts\activate.bat
) else (
    echo [警告] 未检测到虚拟环境，使用系统 Python
    echo [提示] 建议创建虚拟环境：python -m venv venv
)

REM 检查依赖是否安装
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo [信息] 检测到缺少依赖，正在安装...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖安装失败
        pause
        exit /b 1
    )
)

REM 创建必要的目录
if not exist "radarm_data\sessions" mkdir radarm_data\sessions
if not exist "out" mkdir out

echo [信息] 正在启动后端服务...
echo [信息] 服务地址: http://localhost:8000
echo [信息] 按 Ctrl+C 停止服务
echo.

python backend.py

pause

