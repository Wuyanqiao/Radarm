# 流式响应 API 文档

## 概述

Radarm 现在支持 Server-Sent Events (SSE) 流式响应，可以实时展示 Agent 的思考过程和生成内容。

## API 端点

### POST `/chat/stream`

流式聊天接口，返回 SSE 格式的事件流。

**请求格式：**
与 `/chat` 接口相同，使用 `ChatRequest` 模型。

**响应格式：**
Server-Sent Events (text/event-stream)

**事件类型：**

1. **thinking** - 思考过程
   ```json
   {
     "type": "thinking",
     "stage": "planner|executor|verifier|system",
     "content": "🧠 架构师正在规划..."
   }
   ```

2. **content** - 内容块（LLM 生成的内容）
   ```json
   {
     "type": "content",
     "stage": "planner|executor|verifier",
     "content": "文本内容..."
   }
   ```

3. **complete** - 完成事件（包含最终结果）
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

4. **error** - 错误事件
   ```json
   {
     "type": "error",
     "content": "错误信息"
   }
   ```

5. **done** - 流结束标记
   ```json
   {
     "type": "done"
   }
   ```

## 前端使用示例

### JavaScript/TypeScript

```javascript
const eventSource = new EventSource('/chat/stream', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    session_id: 'your-session-id',
    message: '分析数据',
    mode: 'agent_multi',
    apiKeys: {
      deepseekA: 'your-key',
      deepseekB: 'your-key',
      deepseekC: 'your-key'
    }
  })
});

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch (data.type) {
    case 'thinking':
      // 显示思考过程
      console.log(`[${data.stage}] ${data.content}`);
      break;
    
    case 'content':
      // 追加内容
      appendContent(data.content);
      break;
    
    case 'complete':
      // 处理最终结果
      handleComplete(data.data);
      break;
    
    case 'error':
      // 显示错误
      showError(data.content);
      break;
    
    case 'done':
      // 关闭连接
      eventSource.close();
      break;
  }
};

eventSource.onerror = (error) => {
  console.error('SSE 连接错误:', error);
  eventSource.close();
};
```

### 使用 Fetch API (推荐)

```javascript
async function streamChat(message, sessionId, apiKeys) {
  const response = await fetch('/chat/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId,
      message: message,
      mode: 'agent_multi',
      apiKeys: apiKeys
    })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        handleStreamEvent(data);
      }
    }
  }
}

function handleStreamEvent(data) {
  switch (data.type) {
    case 'thinking':
      updateThinkingStatus(data.stage, data.content);
      break;
    case 'content':
      appendToResponse(data.content);
      break;
    case 'complete':
      finalizeResponse(data.data);
      break;
    case 'error':
      showError(data.content);
      break;
  }
}
```

## 当前支持的模式

- ✅ **agent_multi** - 多专家模式（完全支持流式）
- ⏳ **agent_single** - 单模型模式（计划支持）
- ⏳ **ask** - 问答模式（计划支持）

## 优势

1. **实时反馈**：用户可以实时看到 Agent 的思考过程
2. **更好的用户体验**：不需要等待完整响应，可以立即看到进度
3. **降低感知延迟**：即使总时间相同，流式响应让用户感觉更快
4. **调试友好**：可以实时查看每个阶段的输出

## 注意事项

1. **连接管理**：确保正确关闭 SSE 连接，避免资源泄漏
2. **错误处理**：实现完善的错误处理和重连机制
3. **浏览器兼容性**：现代浏览器都支持 SSE，但需要注意 EventSource 的限制
4. **超时处理**：长时间运行的请求可能需要特殊处理

## 技术实现

- **后端**：使用 `httpx.AsyncClient` 进行异步 HTTP 请求
- **流式解析**：解析 OpenAI 兼容的 SSE 格式响应
- **SSE 格式**：遵循 Server-Sent Events 标准
- **异步生成器**：使用 Python `async generator` 实现流式输出

