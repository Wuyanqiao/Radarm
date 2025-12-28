"""
Ask 模式流式服务（简单问答）
"""
import json
from typing import AsyncGenerator, Dict, Any
from backend.app.services.llm_service import LLMService


async def run_ask_stream(
    *,
    user_query: str,
    api_keys: Dict[str, str],
    model_config: Dict[str, Any],
    provider: str,
    model: str,
    llm_service: LLMService,
) -> AsyncGenerator[str, None]:
    """
    Ask 模式流式响应（简单问答，无需代码执行）
    """
    api_key = api_keys.get(provider)
    if not api_key:
        yield json.dumps({"type": "error", "content": f"未配置 {provider} 的 API Key"})
        return
    
    yield json.dumps({
        "type": "thinking",
        "stage": "thinking",
        "content": "🤔 正在思考..."
    })
    
    full_content = ""
    async for chunk in llm_service.call_llm_stream(
        provider=provider,
        api_key=api_key,
        model_config=model_config,
        prompt=user_query,
        model=model,
    ):
        chunk_data = json.loads(chunk)
        if chunk_data.get("type") == "content":
            content = chunk_data.get("content", "")
            full_content += content
            yield json.dumps({
                "type": "content",
                "stage": "response",
                "content": content
            })
        elif chunk_data.get("type") == "complete":
            full_content = chunk_data.get("content", "")
        elif chunk_data.get("type") == "error":
            yield json.dumps({
                "type": "error",
                "content": f"回答生成失败: {chunk_data.get('content', '')}"
            })
            return
    
    # 返回最终结果
    yield json.dumps({
        "type": "complete",
        "data": {
            "reply": full_content,
            "generated_code": None,
            "execution_result": None,
            "image": None,
            "plotly_json": None,
            "new_df": None,
        }
    })

