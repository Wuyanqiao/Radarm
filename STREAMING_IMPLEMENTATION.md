# 流式响应实现文档

## 概述

Radarm 现已全面支持 Server-Sent Events (SSE) 流式响应，可以实时展示 Agent 的思考过程和生成内容。

## 架构设计

### 1. 服务层 (`backend/app/services/`)

#### `llm_service.py` - 异步 LLM 服务
- **连接池优化**：使用类级别的连接池，复用 HTTP 连接
- **流式支持**：`call_llm_stream()` 方法支持流式响应
- **向后兼容**：`call_llm()` 方法支持非流式调用

**关键特性：**
- 连接池：`max_keepalive_connections=10, max_connections=20`
- 超时处理：180秒超时
- 错误处理：完善的异常捕获和错误事件生成

#### `agent_stream_service.py` - 多专家流式服务
- 实时 yield 各阶段思考过程
- 支持 Planner/Executor/Verifier 三阶段流式输出
- 支持多轮迭代和错误反馈

#### `agent_stream_service_single.py` - 单模型流式服务
- 代码生成阶段流式输出
- 代码执行阶段状态更新
- 解释生成阶段流式输出

#### `agent_stream_service_ask.py` - Ask 模式流式服务
- 简单问答的流式响应
- 无需代码执行，直接返回 LLM 响应

### 2. 路由层 (`backend.py`)

#### `/chat/stream` 端点
- 支持三种模式：`agent_multi`, `agent_single`, `ask`
- 返回 SSE 格式 (`text/event-stream`)
- 自动管理 LLM 服务生命周期

### 3. 前端层 (`src/`)

#### `utils/streamChat.js` - 流式聊天工具
- 解析 SSE 格式响应
- 事件分发和处理
- 错误处理和取消支持

#### `App.js` - 前端集成
- 自动检测是否使用流式响应
- 实时更新消息内容
- 显示流式状态指示器

## 事件类型

### thinking - 思考过程
```json
{
  "type": "thinking",
  "stage": "planner|executor|verifier|system|code_generation|execution|explanation",
  "content": "🧠 架构师正在规划..."
}
```

### content - 内容块
```json
{
  "type": "content",
  "stage": "planner|executor|verifier|code_generation|execution|explanation|response",
  "content": "文本内容..."
}
```

### complete - 完成事件
```json
{
  "type": "complete",
  "data": {
    "reply": "最终回复",
    "generated_code": "代码",
    "execution_result": "执行结果",
    "image": "图片路径",
    "plotly_json": "Plotly JSON"
  }
}
```

### error - 错误事件
```json
{
  "type": "error",
  "content": "错误信息"
}
```

### done - 流结束
```json
{
  "type": "done"
}
```

## 性能优化

### 1. 连接池
- **实现**：类级别的 `_client_pool`，所有请求共享
- **配置**：`max_keepalive_connections=10, max_connections=20`
- **优势**：减少连接建立开销，提高并发性能

### 2. 异步处理
- 所有 LLM 调用使用 `httpx.AsyncClient`
- 使用 `async generator` 实现流式输出
- 非阻塞 I/O，提高并发能力

### 3. 资源管理
- 使用 `async with` 确保资源正确释放
- 连接池客户端不关闭，独立客户端自动关闭
- 异常情况下也能正确清理资源

## 测试

### 单元测试 (`tests/test_streaming.py`)
- `test_llm_service_stream()` - 测试流式调用
- `test_llm_service_connection_pool()` - 测试连接池
- `test_llm_service_independent_client()` - 测试独立客户端
- `test_sse_format()` - 测试 SSE 格式
- `test_stream_error_handling()` - 测试错误处理

### 集成测试 (`tests/test_integration_streaming.py`)
- `test_stream_endpoint_exists()` - 测试端点存在性
- `test_stream_response_format()` - 测试响应格式

### 运行测试
```bash
# Windows
run_tests.bat

# Linux/Mac
chmod +x run_tests.sh
./run_tests.sh

# 或使用 pytest
pytest tests/ -v
```

## 使用示例

### 后端调用
```python
from backend.app.services.llm_service import LLMService

async with LLMService() as service:
    async for event in service.call_llm_stream(
        provider="deepseekA",
        api_key="your-key",
        model_config=MODEL_CONFIG,
        prompt="分析数据",
    ):
        data = json.loads(event)
        if data["type"] == "content":
            print(data["content"], end="", flush=True)
```

### 前端调用
```javascript
import { streamChat } from './utils/streamChat';

await streamChat(
  {
    session_id: 'xxx',
    message: '分析数据',
    mode: 'agent_multi',
    apiKeys: { ... }
  },
  (data) => {
    // 处理事件
    if (data.type === 'thinking') {
      console.log(`[${data.stage}] ${data.content}`);
    } else if (data.type === 'content') {
      appendContent(data.content);
    }
  },
  (error) => {
    console.error('Stream error:', error);
  },
  abortController.signal
);
```

## 配置选项

### 前端配置
- `enableStream`: 是否启用流式响应（默认 `true`）
- 可通过会话设置控制是否使用流式

### 后端配置
- `use_pool`: 是否使用连接池（默认 `true`）
- 连接池大小：`max_keepalive_connections=10, max_connections=20`
- 超时时间：180秒

## 故障排查

### 1. 流式响应不工作
- 检查浏览器是否支持 SSE
- 检查网络代理是否缓冲 SSE
- 查看浏览器控制台错误

### 2. 连接池问题
- 检查 `httpx` 版本
- 查看连接数限制
- 检查是否有连接泄漏

### 3. 性能问题
- 检查连接池配置
- 监控并发请求数
- 检查超时设置

## 未来改进

1. **WebSocket 支持**：考虑添加 WebSocket 作为 SSE 的替代
2. **重连机制**：自动重连断开的流式连接
3. **压缩支持**：对 SSE 响应进行压缩
4. **速率限制**：添加请求速率限制
5. **监控指标**：添加性能监控和指标收集

