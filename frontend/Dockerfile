# 前端 Dockerfile - 多阶段构建
# 阶段 1: 构建 React 应用
FROM node:18-alpine AS builder

# 设置工作目录
WORKDIR /app

# 复制 package.json 和 package-lock.json
COPY package*.json ./

# 安装依赖
RUN npm ci

# 复制前端源代码
COPY public ./public
COPY src ./src
COPY tailwind.config.js postcss.config.js ./

# 构建生产版本
RUN npm run build

# 阶段 2: 使用 Nginx 部署
FROM nginx:alpine

# 复制构建产物到 Nginx 静态文件目录
COPY --from=builder /app/build /usr/share/nginx/html

# 复制自定义 Nginx 配置
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# 暴露端口
EXPOSE 80

# 启动 Nginx
CMD ["nginx", "-g", "daemon off;"]

