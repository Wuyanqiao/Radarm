/**
 * 流式聊天工具函数
 */
export async function streamChat(payload, onEvent, onError, signal) {
  const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:8000';
  
  try {
    const response = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal: signal,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      
      if (done) {
        break;
      }

      // 解码数据块
      buffer += decoder.decode(value, { stream: true });
      
      // 按行分割（SSE 格式：data: {...}\n\n）
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // 保留最后一个不完整的行

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            onEvent(data);
          } catch (e) {
            console.warn('Failed to parse SSE data:', line, e);
          }
        }
      }
    }

    // 处理剩余的 buffer
    if (buffer.trim()) {
      const lines = buffer.split('\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            onEvent(data);
          } catch (e) {
            console.warn('Failed to parse SSE data:', line, e);
          }
        }
      }
    }
  } catch (error) {
    if (error.name === 'AbortError') {
      onError(new Error('请求已取消'));
    } else {
      onError(error);
    }
  }
}

