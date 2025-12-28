"""
异步 LLM 服务 - 支持流式响应
"""
import json
import asyncio
from typing import AsyncGenerator, Optional, Dict, Any
import httpx


class LLMService:
    """异步 LLM 调用服务，支持流式响应和连接池"""
    
    # 类级别的连接池（单例模式）
    _client_pool: Optional[httpx.AsyncClient] = None
    _pool_lock = asyncio.Lock()
    
    def __init__(self, use_pool: bool = True):
        """
        初始化 LLM 服务
        
        Args:
            use_pool: 是否使用连接池（默认 True，提高性能）
        """
        self.use_pool = use_pool
        self.client: Optional[httpx.AsyncClient] = None
        self._owns_client = False
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        if self.use_pool:
            # 使用连接池
            async with self._pool_lock:
                if self._client_pool is None:
                    self._client_pool = httpx.AsyncClient(
                        timeout=httpx.Timeout(180.0),
                        limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
                    )
                self.client = self._client_pool
                self._owns_client = False
        else:
            # 创建独立客户端
            self.client = httpx.AsyncClient(timeout=httpx.Timeout(180.0))
            self._owns_client = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        # 只有独立客户端才需要关闭，连接池客户端不关闭
        if self._owns_client and self.client:
            await self.client.aclose()
            self.client = None
    
    async def call_llm_stream(
        self,
        provider: str,
        api_key: str,
        model_config: Dict[str, Any],
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
        timeout: int = 180,
    ) -> AsyncGenerator[str, None]:
        """
        异步流式调用 LLM API
        
        Args:
            provider: 模型提供商（如 'deepseekA', 'zhipu', 'qwen'）
            api_key: API Key
            model_config: 模型配置字典
            prompt: 提示词
            model: 可选，覆盖默认模型
            temperature: 温度参数
            timeout: 超时时间（秒）
        
        Yields:
            str: 流式返回的文本块
        """
        if not api_key:
            yield json.dumps({"type": "error", "content": f"缺少 {provider} 的 API Key"})
            return
        
        cfg = model_config.get(provider)
        if not cfg:
            yield json.dumps({"type": "error", "content": f"未知模型 provider={provider}"})
            return
        
        if not self.client:
            self.client = httpx.AsyncClient(timeout=httpx.Timeout(float(timeout)))
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": model or cfg.get("model"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": True  # 启用流式响应
        }
        
        url = cfg.get("url")
        if not url:
            yield json.dumps({"type": "error", "content": f"缺少 {provider} 的 URL 配置"})
            return
        
        try:
            async with self.client.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
                timeout=timeout
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield json.dumps({
                        "type": "error",
                        "content": f"API 调用失败: {response.status_code} {error_text.decode('utf-8', errors='ignore')}"
                    })
                    return
                
                buffer = ""
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    # SSE 格式：data: {...} 或直接 JSON
                    if line.startswith("data: "):
                        data_str = line[6:]  # 移除 "data: " 前缀
                    else:
                        data_str = line.strip()
                    
                    if not data_str:
                        continue
                    
                    # OpenAI 格式的流式响应结束标记
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                buffer += content
                                yield json.dumps({
                                    "type": "content",
                                    "content": content
                                })
                    except json.JSONDecodeError:
                        # 某些 API 可能返回非 JSON 格式，跳过
                        continue
                
                # 返回完整内容（用于后续处理）
                if buffer:
                    yield json.dumps({
                        "type": "complete",
                        "content": buffer
                    })
        
        except httpx.TimeoutException:
            yield json.dumps({
                "type": "error",
                "content": f"请求超时（{timeout}秒）"
            })
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "content": f"请求异常: {str(e)}"
            })
    
    async def call_llm(
        self,
        provider: str,
        api_key: str,
        model_config: Dict[str, Any],
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
        timeout: int = 180,
    ) -> Optional[str]:
        """
        异步非流式调用 LLM API（向后兼容）
        
        Returns:
            str: 完整的响应文本，失败返回 None
        """
        full_content = ""
        async for chunk in self.call_llm_stream(
            provider=provider,
            api_key=api_key,
            model_config=model_config,
            prompt=prompt,
            model=model,
            temperature=temperature,
            timeout=timeout,
        ):
            try:
                data = json.loads(chunk)
                if data.get("type") == "content":
                    full_content += data.get("content", "")
                elif data.get("type") == "error":
                    return None
            except json.JSONDecodeError:
                continue
        
        return full_content if full_content else None


# 全局服务实例（单例模式）
_llm_service: Optional[LLMService] = None


async def get_llm_service() -> LLMService:
    """获取 LLM 服务实例"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
        await _llm_service.__aenter__()
    return _llm_service


async def close_llm_service():
    """关闭 LLM 服务"""
    global _llm_service
    if _llm_service:
        await _llm_service.__aexit__(None, None, None)
        _llm_service = None

