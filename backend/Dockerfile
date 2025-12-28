# 后端 Dockerfile
# 基于 Python 3.10 slim 镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（包括中文字体）
RUN apt-get update && apt-get install -y \
    # 中文字体支持
    fonts-wqy-zenhei \
    fonts-wqy-microhei \
    # Tesseract OCR 依赖
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    # 编译依赖（某些 Python 包需要）
    gcc \
    g++ \
    # 其他系统工具
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FOR_DISABLE_CONSOLE_CTRL_HANDLER=1

# 复制 requirements.txt 并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 创建必要的目录（包括 engines，即使它可能为空）
RUN mkdir -p out radarm_data/sessions engines

# 复制后端 Python 文件（排除前端和构建产物）
# 注意：build context 是项目根目录，所以这里复制根目录下的 .py 文件
COPY *.py ./

# 复制 engines 目录（如果存在且有内容）
# 注意：如果 engines 目录不存在，这个 COPY 命令会失败
# 由于 engines 目录当前为空，我们跳过复制，目录已在上面创建
# COPY engines/ ./engines/

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "backend.py"]

