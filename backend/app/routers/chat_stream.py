"""
流式聊天路由 - 直接集成到 backend.py
避免循环导入，直接使用 backend.py 中的函数和变量
"""
import json
import asyncio
from typing import AsyncGenerator
from fastapi.responses import StreamingResponse

# 注意：这个文件中的函数会被 backend.py 导入使用
# 避免在这里导入 backend.py 的内容，而是通过参数传递


async def generate_chat_stream(
    req,
    session,
    current_df,
    data_context: str,
    mode: str,
    api_keys: dict,
    model_config: dict,
    roles: dict,
    execute_callback,
    llm_service,
    # 导入的函数
    run_multi_agent_engine_stream_func,
) -> AsyncGenerator[str, None]:
    """
    生成流式聊天响应
    
    返回 SSE 格式的事件流
    """
    try:
        # 流式执行多专家引擎
        async for event in run_multi_agent_engine_stream_func(
            user_query=req.message,
            data_context=data_context,
            api_keys=api_keys,
            model_config=model_config,
            roles=roles,
            execute_callback=execute_callback,
            df=current_df,
            llm_service=llm_service,
        ):
            # 转换为 SSE 格式
            yield f"data: {event}\n\n"
        
        # 发送完成事件
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': f'处理失败: {str(e)}'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

