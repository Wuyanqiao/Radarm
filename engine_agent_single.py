"""
单模型 Agent 引擎（底层能力）
----------------------------
用于“用户聊天 Radarm AI agent 的单模型模式”。

职责：
1) 单模型生成 Python 代码（面向 DataFrame `df`）
2) 通过后端提供的 execute_callback(code, df) 在沙盒执行
3) 再次调用同一模型生成简要解释

不负责：
- FastAPI 路由与 Session 状态（见 backend.py）
"""

import re
import importlib
from typing import Any, Dict, Optional, List

PROMPT_TEMPLATE = """
你是 Python 数据分析专家。
运行环境中已有 DataFrame `df`（内存数据），严禁读取任何外部文件/网络（不要 read_csv/read_excel，不要 data.csv，不要 requests）。
用户需求：{user_query}
数据概况：{data_context}

【重要提示 - 图片和视觉理解数据】
如果上面的"数据概况"中包含"[视觉理解]"或"[图片附件]"部分：
1. **完整理解图片信息**：仔细阅读视觉理解结果，理解图片中的所有信息（文字、表格、图表、标准、规范、界面元素、图像内容等）
2. **提取并使用图片信息**：根据用户需求和视觉理解结果，提取图片中的任何相关信息并在代码中使用
3. **结构化数据定义**：如果图片包含表格、标准、规范、限值等结构化信息，且代码中需要使用这些信息，**必须在代码开头先解析并定义相应的数据结构**（如字典、DataFrame、列表等）
4. **示例**：
   - 如果视觉理解提到标准限值（如"总酸≥0.4（优级）"），应创建类似 `standards = {{'总酸': {{'优级': 0.4, '一级': 0.3}}}}` 的结构
   - 如果视觉理解提到表格数据，应创建相应的DataFrame或字典结构
   - 如果视觉理解提到其他结构化信息，应根据需要创建相应的数据结构
5. **避免硬编码**：确保代码中使用的图片信息都从视觉理解结果中提取并定义，而不是直接硬编码或引用未定义的变量
6. **充分利用所有信息**：不要只关注表格或标准，要充分利用图片中的任何相关信息（文字说明、图表趋势、界面状态等）

【规则】
1. 机器学习任务必须调用 `ml.run(df, ...)`。
2. 普通分析可用 pandas/numpy/matplotlib/seaborn。
3. 关键结论必须赋值给变量 `result`（字符串或数值均可）。
4. 绘图不要调用 `plt.show()`。
5. 只输出一个 ```python 代码块，不要解释。
"""


def _call_llm(provider: str, api_key: str, model_config: Dict[str, Any], prompt: str) -> Optional[str]:
    if not api_key:
        return None
    cfg = model_config.get(provider)
    if not cfg:
        return None

    try:
        requests = importlib.import_module("requests")
    except Exception:
        return None

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {"model": cfg["model"], "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
    try:
        resp = requests.post(cfg["url"], headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            return None
        return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def _extract_python_code(ai_response: str) -> str:
    m = re.search(r"```python(.*?)```", ai_response, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ai_response.strip()

def _auto_feedback_from_exec(exec_text: Any) -> str:
    t = str(exec_text or "")
    if not t:
        return ""
    if "No such file or directory" in t or "FileNotFoundError" in t:
        return "不要读取任何本地文件（如 data.csv）。运行环境已提供 df，请直接使用 df 进行分析。"
    if "KeyError" in t:
        return "出现 KeyError（列名不存在）。请检查 df.columns，必要时做列名模糊匹配，并在代码中处理列不存在的情况（给出明确提示）。"
    if "禁止文件/网络/系统操作" in t:
        return "系统禁止文件/网络/系统操作。请移除 read_csv/read_excel/open/requests 等，直接使用 df 进行计算。"
    return ""


def run_single_agent_engine(
    *,
    user_query: str,
    data_context: str = "",
    api_keys: Dict[str, str],
    primary_model: str,
    model_config: Dict[str, Any],
    execute_callback,
    df,
) -> Dict[str, Any]:
    """
    单模型 Agent 引擎入口（供 workflow_single_chat.py 调用）
    """
    api_key = api_keys.get(primary_model)
    if not api_key:
        return {"error": f"未配置 {primary_model} 的 API Key", "process_log": f"❌ missing_key provider={primary_model}"}

    process_log: List[str] = []
    max_attempts = 2
    code = ""
    exec_text, exec_img, plotly_json, new_df = "", None, None, None

    for attempt in range(1, max_attempts + 1):
        # 1) 生成代码（注入数据概况）
        prompt = PROMPT_TEMPLATE.format(user_query=user_query, data_context=(data_context or ""))
        if attempt > 1:
            prompt = (
                prompt
                + "\n\n【上一轮执行失败反馈】\n"
                + (_auto_feedback_from_exec(exec_text) or "请根据报错修复代码，并确保 result 有输出。")
                + "\n\n【上一轮错误】\n"
                + (str(exec_text)[:2000] + ("...(截断)" if len(str(exec_text)) > 2000 else ""))
                + "\n\n请输出修复后的完整 Python 代码块（```python ...```），务必可执行。"
            )

        process_log.append(f"#### [单模型] 第 {attempt} 次生成代码（{primary_model}）")
        ai_response = _call_llm(primary_model, api_key, model_config, prompt)
        if not ai_response:
            return {"error": "AI 无响应或请求超时", "process_log": "\n".join(process_log)}

        # 2) 执行代码
        code = _extract_python_code(ai_response)
        process_log.append("**⚙️ 执行代码...**")
        # 支持新的4元组返回： (output_text, image_path, plotly_json, new_df)
        result = execute_callback(code, df)
        if len(result) == 4:
            exec_text, exec_img, plotly_json, new_df = result
        else:
            # 向后兼容：如果是3元组，添加 None 作为 plotly_json
            exec_text, exec_img, new_df = result
            plotly_json = None

        has_error = isinstance(exec_text, str) and (
            exec_text.startswith("Error") or "Traceback" in exec_text or "Exception" in exec_text
        )
        if not has_error:
            break
        process_log.append(f"⚠️ **报错**: {str(exec_text)[:200]}")
        if attempt < max_attempts:
            fb = _auto_feedback_from_exec(exec_text)
            if fb:
                process_log.append(f"🤖 **系统提示**: {fb}")

    # 3) 生成解释（避免把超长执行输出塞回模型）
    exec_text_for_explain = exec_text
    if isinstance(exec_text_for_explain, str) and len(exec_text_for_explain) > 2000:
        exec_text_for_explain = exec_text_for_explain[:2000] + "...(截断)"
    explain_prompt = (
        f"用户需求：{user_query}\n"
        f"数据概况：{(data_context or '')[:6000]}\n\n"
        f"代码执行结果：{exec_text_for_explain}\n\n"
        "请用中文简要解释分析结果，并给出下一步建议（不要杜撰数据）。"
    )
    explain_res = _call_llm(primary_model, api_key, model_config, explain_prompt)

    return {
        "reply": explain_res if explain_res else "执行完成（AI未返回解释）",
        "generated_code": code,
        "execution_result": exec_text,
        "image": exec_img,
        "plotly_json": plotly_json,  # 新增：Plotly 图表 JSON
        "new_df": new_df,
        "process_log": "\n".join(process_log),
    }


