"""
单模型 Agent 流式服务
"""
import json
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
from backend.app.services.llm_service import LLMService


async def run_single_agent_engine_stream(
    *,
    user_query: str,
    data_context: str,
    api_keys: Dict[str, str],
    model_config: Dict[str, Any],
    primary_model: str,
    execute_callback,
    df,
    llm_service: LLMService,
) -> AsyncGenerator[str, None]:
    """
    单模型 Agent 引擎 - 流式版本
    
    实时 yield 思考过程：
    - "thinking: 正在生成代码..."
    - "thinking: 正在运行代码..."
    - "content: <实际内容>"
    - "complete: <最终结果>"
    """
    from engine_agent_single import (
        PROMPT_TEMPLATE,
        _extract_python_code,
        _auto_feedback_from_exec,
    )
    
    api_key = api_keys.get(primary_model)
    if not api_key:
        yield json.dumps({"type": "error", "content": f"未配置 {primary_model} 的 API Key"})
        return
    
    max_attempts = 2
    code = ""
    exec_text = ""
    exec_img = None
    plotly_json = None
    new_df = df
    
    for attempt in range(1, max_attempts + 1):
        # 1) 生成代码
        yield json.dumps({
            "type": "thinking",
            "stage": "code_generation",
            "content": f"💻 正在生成代码（第 {attempt} 次尝试）..."
        })
        
        prompt = PROMPT_TEMPLATE.format(
            user_query=user_query,
            data_context=(data_context or "")
        )
        
        if attempt > 1:
            prompt = (
                prompt
                + "\n\n【上一轮执行失败反馈】\n"
                + (_auto_feedback_from_exec(exec_text) or "请根据报错修复代码，并确保 result 有输出。")
                + "\n\n【上一轮错误】\n"
                + (str(exec_text)[:2000] + ("...(截断)" if len(str(exec_text)) > 2000 else ""))
                + "\n\n请输出修复后的完整 Python 代码块（```python ...```），务必可执行。"
            )
        
        code_res = ""
        async for chunk in llm_service.call_llm_stream(
            provider=primary_model,
            api_key=api_key,
            model_config=model_config,
            prompt=prompt,
        ):
            chunk_data = json.loads(chunk)
            if chunk_data.get("type") == "content":
                code_res += chunk_data.get("content", "")
                yield json.dumps({
                    "type": "content",
                    "stage": "code_generation",
                    "content": chunk_data.get("content", "")
                })
            elif chunk_data.get("type") == "complete":
                code_res = chunk_data.get("content", "")
            elif chunk_data.get("type") == "error":
                yield json.dumps({
                    "type": "error",
                    "content": f"代码生成失败: {chunk_data.get('content', '')}"
                })
                return
        
        code = _extract_python_code(code_res)
        
        # 2) 执行代码
        yield json.dumps({
            "type": "thinking",
            "stage": "execution",
            "content": "⚙️ 正在执行代码..."
        })
        
        result = execute_callback(code, df)
        if len(result) == 4:
            exec_text, exec_img, plotly_json, new_df = result
        else:
            exec_text, exec_img, new_df = result
            plotly_json = None
        
        has_error = isinstance(exec_text, str) and (
            exec_text.startswith("Error") or "Traceback" in exec_text or "Exception" in exec_text
        )
        
        if not has_error:
            break
        
        yield json.dumps({
            "type": "thinking",
            "stage": "execution",
            "content": f"⚠️ 执行报错: {str(exec_text)[:100]}..."
        })
        
        if attempt < max_attempts:
            yield json.dumps({
                "type": "thinking",
                "stage": "retry",
                "content": f"🔄 准备重试（{attempt + 1}/{max_attempts}）..."
            })
    
    # 3) 生成解释
    yield json.dumps({
        "type": "thinking",
        "stage": "explanation",
        "content": "📝 正在生成解释..."
    })
    
    exec_text_for_explain = exec_text
    if isinstance(exec_text_for_explain, str) and len(exec_text_for_explain) > 2000:
        exec_text_for_explain = exec_text_for_explain[:2000] + "...(截断)"
    
    explain_prompt = (
        f"用户需求：{user_query}\n"
        f"数据概况：{(data_context or '')[:6000]}\n\n"
        f"代码执行结果：{exec_text_for_explain}\n\n"
        "请用中文简要解释分析结果，并给出下一步建议（不要杜撰数据）。"
    )
    
    explain_res = ""
    async for chunk in llm_service.call_llm_stream(
        provider=primary_model,
        api_key=api_key,
        model_config=model_config,
        prompt=explain_prompt,
    ):
        chunk_data = json.loads(chunk)
        if chunk_data.get("type") == "content":
            explain_res += chunk_data.get("content", "")
            yield json.dumps({
                "type": "content",
                "stage": "explanation",
                "content": chunk_data.get("content", "")
            })
        elif chunk_data.get("type") == "complete":
            explain_res = chunk_data.get("content", "")
        elif chunk_data.get("type") == "error":
            explain_res = "解释生成失败"
    
    # 返回最终结果
    yield json.dumps({
        "type": "complete",
        "data": {
            "reply": explain_res if explain_res else "执行完成（AI未返回解释）",
            "generated_code": code,
            "execution_result": exec_text,
            "image": exec_img,
            "plotly_json": plotly_json,
            "new_df": None,
        }
    })

