"""
Agent 流式服务 - 支持实时输出思考过程
"""
import json
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
from backend.app.services.llm_service import LLMService


async def run_multi_agent_engine_stream(
    *,
    user_query: str,
    data_context: str,
    api_keys: Dict[str, str],
    model_config: Dict[str, Any],
    roles: Optional[Dict[str, str]] = None,
    execute_callback,
    df,
    llm_service: LLMService,
) -> AsyncGenerator[str, None]:
    """
    多专家混合 Agent 引擎 - 流式版本
    
    实时 yield 思考过程：
    - "thinking: 正在规划..."
    - "thinking: 正在生成代码..."
    - "thinking: 正在运行代码..."
    - "thinking: 正在审核..."
    - "content: <实际内容>"
    - "complete: <最终结果>"
    """
    import re
    # 导入必要的函数（避免循环导入）
    from engine_agent_multi import (
        PROMPTS,
        _extract_python_code,
        _extract_json,
        _auto_feedback_from_exec,
        _provider_label,
        build_semantic_hints,
    )
    
    roles = roles or {"planner": "deepseekA", "executor": "deepseekB", "verifier": "deepseekC"}
    
    # Key 检查与自动补位
    available_keys = [k for k, v in api_keys.items() if v]
    if not available_keys:
        yield json.dumps({"type": "error", "content": "未配置 API Key"})
        return
    
    for r in roles:
        if not api_keys.get(roles[r]):
            roles[r] = available_keys[0]
    
    iteration = 0
    max_iterations = 2
    feedback = ""
    exec_text = ""
    exec_img = None
    plotly_json = None
    new_df = df
    code = ""
    
    while iteration < max_iterations:
        iter_prefix = f"第 {iteration + 1} 轮迭代"
        feedback_context = f"\n\n[上一轮反馈]\n{feedback}\n" if feedback else ""
        
        # 构建数据上下文
        try:
            hints = build_semantic_hints([str(c) for c in getattr(df, "columns", [])])
            hints_text = json.dumps(hints, ensure_ascii=False, indent=2)
        except Exception:
            hints_text = "{}"
        
        enriched_data_context = (
            (data_context or "")
            + "\n\n[字段候选映射(JSON)]\n"
            + hints_text
            + "\n\n请优先使用上述 core_columns/qc_covariates 中的真实列名；若用户提到的概念无对应列，必须在结论中说明并给出补充字段建议。"
        )
        
        # 1) 规划阶段
        yield json.dumps({
            "type": "thinking",
            "stage": "planner",
            "content": f"🧠 架构师 ({_provider_label(roles['planner'])}) 正在规划..."
        })
        
        plan = ""
        async for chunk in llm_service.call_llm_stream(
            provider=roles["planner"],
            api_key=api_keys[roles["planner"]],
            model_config=model_config,
            prompt=PROMPTS["planner"].format(
                user_query=user_query,
                data_context=enriched_data_context,
                feedback_context=feedback_context
            ),
        ):
            chunk_data = json.loads(chunk)
            if chunk_data.get("type") == "content":
                plan += chunk_data.get("content", "")
                yield json.dumps({
                    "type": "content",
                    "stage": "planner",
                    "content": chunk_data.get("content", "")
                })
            elif chunk_data.get("type") == "complete":
                plan = chunk_data.get("content", "")
            elif chunk_data.get("type") == "error":
                yield json.dumps({
                    "type": "error",
                    "content": f"规划阶段失败: {chunk_data.get('content', '')}"
                })
                return
        
        if plan.startswith("Error"):
            yield json.dumps({"type": "error", "content": plan})
            return
        
        yield json.dumps({
            "type": "thinking",
            "stage": "planner",
            "content": f"✅ 规划完成：{plan[:100]}..."
        })
        
        # 2) 执行阶段
        yield json.dumps({
            "type": "thinking",
            "stage": "executor",
            "content": f"💻 程序员 ({_provider_label(roles['executor'])}) 正在编码..."
        })
        
        code_res = ""
        async for chunk in llm_service.call_llm_stream(
            provider=roles["executor"],
            api_key=api_keys[roles["executor"]],
            model_config=model_config,
            prompt=PROMPTS["executor"].format(
                plan=plan,
                data_context=enriched_data_context
            ),
        ):
            chunk_data = json.loads(chunk)
            if chunk_data.get("type") == "content":
                code_res += chunk_data.get("content", "")
                yield json.dumps({
                    "type": "content",
                    "stage": "executor",
                    "content": chunk_data.get("content", "")
                })
            elif chunk_data.get("type") == "complete":
                code_res = chunk_data.get("content", "")
            elif chunk_data.get("type") == "error":
                yield json.dumps({
                    "type": "error",
                    "content": f"编码阶段失败: {chunk_data.get('content', '')}"
                })
                return
        
        code = _extract_python_code(code_res)
        
        # 3) 运行代码
        yield json.dumps({
            "type": "thinking",
            "stage": "executor",
            "content": "⚙️ 正在运行代码..."
        })
        
        result = execute_callback(code, df)
        if len(result) == 4:
            exec_text, exec_img, plotly_json, new_df = result
        else:
            exec_text, exec_img, new_df = result
            plotly_json = None
        
        has_error = isinstance(exec_text, str) and (
            exec_text.startswith("Error") or "Traceback" in exec_text
        )
        
        if has_error:
            yield json.dumps({
                "type": "thinking",
                "stage": "executor",
                "content": f"⚠️ 代码执行报错: {exec_text[:100]}..."
            })
            
            # 自动反馈
            auto_fb = _auto_feedback_from_exec(exec_text)
            if auto_fb:
                yield json.dumps({
                    "type": "thinking",
                    "stage": "system",
                    "content": f"🤖 系统自动诊断: {auto_fb}"
                })
                feedback = auto_fb
                iteration += 1
                continue
        
        # 4) 验证阶段
        yield json.dumps({
            "type": "thinking",
            "stage": "verifier",
            "content": f"⚖️ 评审员 ({_provider_label(roles['verifier'])}) 正在审核..."
        })
        
        force_fail = "\n\n⚠️ 代码报错，请判 FAIL 并说明原因与修复建议！" if has_error else ""
        verify_res = ""
        async for chunk in llm_service.call_llm_stream(
            provider=roles["verifier"],
            api_key=api_keys[roles["verifier"]],
            model_config=model_config,
            prompt=PROMPTS["verifier"].format(
                plan=plan,
                code=code,
                execution_result=exec_text
            ) + force_fail,
        ):
            chunk_data = json.loads(chunk)
            if chunk_data.get("type") == "content":
                verify_res += chunk_data.get("content", "")
            elif chunk_data.get("type") == "complete":
                verify_res = chunk_data.get("content", "")
            elif chunk_data.get("type") == "error":
                yield json.dumps({
                    "type": "error",
                    "content": f"审核阶段失败: {chunk_data.get('content', '')}"
                })
                return
        
        review = _extract_json(verify_res)
        if not isinstance(review, dict):
            review = {}
        
        status = str(review.get("status") or "").upper()
        reason = str(review.get("reason") or "").strip() or "评审未给出明确原因"
        suggestion = str(review.get("suggestion") or "").strip()
        final_reply = str(review.get("final_reply") or "").strip()
        
        if status not in ("PASS", "FAIL"):
            status = "FAIL" if has_error else "FAIL"
            if not suggestion:
                suggestion = "评审输出无法解析为合法 JSON。请严格按 JSON Schema 输出，并修复代码/结果为空等问题。"
        
        if status == "PASS":
            yield json.dumps({
                "type": "thinking",
                "stage": "verifier",
                "content": f"✅ 验证通过: {reason}"
            })
            
            # 返回最终结果
            yield json.dumps({
                "type": "complete",
                "data": {
                    "reply": final_reply if final_reply else f"### 🎯 Radarm 多专家结论\n\n**结论**: {exec_text}\n\n**评审**: {reason}",
                    "generated_code": code,
                    "execution_result": exec_text,
                    "image": exec_img,
                    "plotly_json": plotly_json,
                    "new_df": None,  # DataFrame 不能直接序列化，需要单独处理
                }
            })
            return
        
        yield json.dumps({
            "type": "thinking",
            "stage": "verifier",
            "content": f"❌ 驳回: {reason}\n🔄 建议: {suggestion}"
        })
        
        feedback = suggestion
        iteration += 1
    
    # 达到最大迭代次数
    yield json.dumps({
        "type": "complete",
        "data": {
            "reply": f"⚠️ 达到最大迭代次数。最后结果: {exec_text}",
            "generated_code": code,
            "execution_result": exec_text,
            "image": exec_img,
            "plotly_json": plotly_json,
            "new_df": None,
        }
    })

