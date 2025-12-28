"""
流式聊天路由 - SSE (Server-Sent Events)
"""
import json
import asyncio
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd

from backend.app.services.llm_service import LLMService, get_llm_service
from backend.app.services.agent_stream_service import run_multi_agent_engine_stream
from backend.app.utils.execute_code import execute_code

router = APIRouter(prefix="/api/v1", tags=["chat"])


class ChatStreamRequest(BaseModel):
    """流式聊天请求"""
    session_id: str
    message: str
    mode: str = "agent_single"  # agent_single, agent_multi, ask
    apiKeys: dict = {}
    modelSelection: dict = {}
    agentRoles: dict = {}
    dataContext: str = ""
    webSearch: bool = False
    visionEnabled: bool = True
    imagePaths: list = []


# 导入必要的函数和配置
# 注意：这些需要从 backend.py 导入，但为了避免循环导入，我们通过参数传递
async def chat_stream_handler(
    req: ChatStreamRequest,
    sessions: dict,
    MODEL_CONFIG: dict,
    get_session_data_func,
    update_session_data_func,
    sanitize_df_for_json_func,
    _history_info_func,
    _history_stack_func,
    _meta_current_func,
) -> AsyncGenerator[str, None]:
    """
    流式聊天处理函数
    
    返回 SSE 格式的事件流
    """
    session = get_session_data_func(req.session_id)
    current_df = session['df']
    
    # 统一模式名
    mode = req.mode.lower()
    if mode not in ("agent_single", "agent_multi", "ask"):
        mode = "agent_single"
    
    # 只支持 agent_multi 的流式输出（其他模式可以后续添加）
    if mode != "agent_multi":
        yield f"data: {json.dumps({'type': 'error', 'content': '当前仅支持 agent_multi 模式的流式响应'})}\n\n"
        return
    
    # 准备数据上下文
    data_context = req.dataContext or session.get("profile_context", "")
    
    # 准备 API Keys
    api_keys = req.apiKeys or {}
    
    # 准备角色配置
    roles_in = req.agentRoles if isinstance(req.agentRoles, dict) else {}
    roles = {"planner": "deepseekA", "executor": "deepseekB", "verifier": "deepseekC"}
    overrides = {}
    for r in ("planner", "executor", "verifier"):
        rc = roles_in.get(r) if isinstance(roles_in.get(r), dict) else {}
        prov = str(rc.get("provider") or roles[r])
        roles[r] = prov
        if rc.get("model"):
            overrides[prov] = str(rc.get("model"))
    
    # 应用模型覆盖
    model_config = MODEL_CONFIG.copy()
    for k, v in overrides.items():
        if k in model_config:
            model_config[k] = {**model_config[k], "model": v}
    
    # 执行回调
    def execute_with_session(code_str: str, df: pd.DataFrame):
        return execute_code(code_str, df, session_id=req.session_id)
    
    # 获取 LLM 服务
    llm_service = await get_llm_service()
    
    try:
        # 流式执行多专家引擎
        async for event in run_multi_agent_engine_stream(
            user_query=req.message,
            data_context=data_context,
            api_keys=api_keys,
            model_config=model_config,
            roles=roles,
            execute_callback=execute_with_session,
            df=current_df,
            llm_service=llm_service,
        ):
            # 转换为 SSE 格式
            yield f"data: {event}\n\n"
        
        # 获取最终数据框（需要单独处理，因为不能序列化）
        # 这里我们需要在执行后更新 session
        # 注意：在实际实现中，需要从 execute_callback 的返回值中获取 new_df
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': f'处理失败: {str(e)}'})}\n\n"
    finally:
        # 发送完成事件
        yield f"data: {json.dumps({'type': 'done'})}\n\n"


@router.post("/chat/stream")
async def chat_stream(
    req: ChatStreamRequest,
    sessions: dict = None,  # 需要通过依赖注入传递
    MODEL_CONFIG: dict = None,
    # ... 其他依赖
):
    """
    流式聊天接口 - SSE
    
    返回 Server-Sent Events 流，实时推送：
    - thinking: 思考过程
    - content: 内容块
    - complete: 完成事件
    - error: 错误事件
    """
    # 注意：这里需要通过依赖注入获取 sessions 和 MODEL_CONFIG
    # 为了简化，我们暂时通过参数传递
    # 实际使用时应该通过 FastAPI 的 Depends 注入
    
    async def generate():
        async for event in chat_stream_handler(
            req=req,
            sessions=sessions or {},
            MODEL_CONFIG=MODEL_CONFIG or {},
            get_session_data_func=None,  # 需要通过依赖注入
            update_session_data_func=None,
            sanitize_df_for_json_func=None,
            _history_info_func=None,
            _history_stack_func=None,
            _meta_current_func=None,
        ):
            yield event
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        }
    )

