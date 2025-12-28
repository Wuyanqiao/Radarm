"""
[已弃用] 历史版本的多专家引擎实现（保留作参考）

当前项目的 AI 调度已拆分为 3 个独立底层引擎文件：
- engine_report.py：五阶段报告生成引擎（多专家混合-报告版）
- engine_agent_single.py：单模型 Agent 引擎
- engine_agent_multi.py：多专家混合 Agent 引擎

注意：backend.py 当前不再引用本文件。
"""

import requests
import re
import json
import time

# --- 辅助工具：鲁棒的 JSON 解析器 ---
def extract_and_parse_json(text):
    """
    从 LLM 的回复中提取并解析 JSON。
    支持处理 ```json ... ``` 包裹的情况，以及不规范的格式。
    """
    try:
        # 1. 尝试直接解析
        return json.loads(text)
    except:
        pass

    try:
        # 2. 尝试提取代码块中的 JSON
        match = re.search(r"```json(.*?)```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip())
        
        # 3. 尝试提取大括号 {} 之间的内容
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0).strip())
    except:
        pass
    
    return None

# --- 提示词库 (Radarm Engine) ---
PROMPTS = {
    "planner": """
    你是一名【数学建模架构师】。
    用户问题：{user_query}
    数据概况：{data_context}
    {feedback_context}
    
    请输出详细的分析蓝图。
    要求：
    1. 逻辑严密，分为：预处理 -> 模型选择 -> 求解 -> 验证。
    2. 明确每一步用到的具体算法（如：使用随机森林填补缺失值，使用 ARIMA 预测）。
    3. 不要写代码，只写计划。
    """,

    "executor": """
    你是一名【建模程序员】。
    
    【架构师蓝图】
    {plan}
    
    【数据概况】
    {data_context}
    
    【工具箱】
    1. pandas (pd), numpy (np), matplotlib.pyplot (plt), seaborn (sns)
    2. 机器学习: `ml.run(df, target='...', task='regression'/'classification'/'clustering', k=...)`
    
    【任务】
    编写 Python 代码实现蓝图。
    1. 必须将最终结论赋值给变量 `result`。
    2. 绘图不要使用 `plt.show()`。
    3. 注意数据类型，遇到 NaN 请先处理。
    
    仅输出 ```python 代码块。
    """,

    "verifier": """
    你是一名【建模评审】（铁面无私）。
    
    【架构师蓝图】
    {plan}
    
    【程序员代码】
    {code}
    
    【运行结果】
    {execution_result}
    
    【审查红线 - 违反任一条必须判 FAIL】
    1. ❌ **代码报错**：结果中包含 "Error", "Traceback", "Exception"。
    2. ❌ **结果为空**：结果是 "None" 或空字符串。
    3. ❌ **图表缺失**：如果蓝图要求画图但代码没画（未调用 plt）。
    4. ❌ **偏离蓝图**：未实现蓝图中的核心算法。
    
    请输出标准 JSON（严禁 Markdown）：
    {{
        "status": "PASS" 或 "FAIL",
        "reason": "通过的理由或失败的具体原因（如：第N行代码报错）",
        "suggestion": "给程序员的具体修复建议（如果是报错，请提供修复后的代码片段思路）"
    }}
    """
}

REPORT_PROMPTS = {
    "architect": """
    你是一名【资深学术顾问】。
    数据信息：{data_info}
    
    请设计一份《数学建模分析报告》的大纲。
    包含：摘要、问题重述、数据预处理、模型假设、建模求解、结论与建议。
    输出格式：Markdown 列表。
    """,

    "writer": """
    你是一名【专业论文撰写人】。
    大纲：
    {outline}
    
    统计摘要：
    {data_desc}
    
    【审稿人反馈（如果有）】
    {feedback}
    
    请撰写/修改分析报告。
    要求：
    1. 学术严谨，Markdown 格式。
    2. 必须引用统计摘要中的具体数字（如均值、相关系数）。
    3. 语言客观，避免口语。
    """,

    "reviewer": """
    你是一名【学术期刊审稿人】。
    
    【待审阅稿件】
    {draft}
    
    请严格评审这份报告。
    1. 是否包含具体数据支持？
    2. 逻辑是否通顺？
    3. 格式是否规范？
    
    请输出标准 JSON（严禁 Markdown）：
    {{
        "status": "PASS" 或 "FAIL",
        "comments": "详细的评审意见",
        "revised_content": "如果只有小问题，请直接提供润色后的全文；如果问题严重判FAIL，此字段留空。"
    }}
    """
}

def call_agent_llm(provider, api_key, model_config, prompt):
    if not api_key: return f"Error: 缺少 {provider} 的 API Key"
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model_config[provider]["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1 # 保持低温度以获得稳定输出
    }
    try:
        resp = requests.post(model_config[provider]["url"], headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            return f"Error: API调用失败 ({resp.status_code}) - {resp.text}"
        return resp.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: 请求异常 - {str(e)}"

# --- 核心 1：求解闭环 (Chat Mode) ---
def run_expert_loop(user_query, data_context, api_keys, model_config, execute_callback, df):
    roles = {"planner": "deepseek", "executor": "qwen", "verifier": "zhipu"}
    
    available_keys = [k for k, v in api_keys.items() if v]
    if not available_keys: return {"error": "未配置任何 API Key", "process_log": "❌ 无 Key"}
    for r in roles: 
        if not api_keys.get(roles[r]): roles[r] = available_keys[0]

    process_log = []
    iteration = 0
    max_iterations = 3 # 允许最多3次尝试
    feedback = ""

    while iteration < max_iterations:
        iter_prefix = f"#### [第 {iteration + 1} 轮迭代]"
        
        # 1. 规划
        if iteration == 0:
            # 首轮规划
            process_log.append(f"{iter_prefix}\n**🧠 架构师 ({roles['planner']}) 正在规划...**")
            plan_prompt = PROMPTS["planner"].format(user_query=user_query, data_context=data_context, feedback_context="")
        else:
            # 基于反馈重新规划 (或者跳过规划直接修代码，这里简化为重新规划以确保一致性)
            process_log.append(f"{iter_prefix}\n**🧠 架构师 ({roles['planner']}) 根据反馈调整蓝图...**")
            plan_prompt = PROMPTS["planner"].format(user_query=user_query, data_context=data_context, feedback_context=f"【上一轮失败原因】：{feedback}")

        plan = call_agent_llm(roles["planner"], api_keys[roles["planner"]], model_config, plan_prompt)
        if plan.startswith("Error"): return {"error": plan, "process_log": "\n".join(process_log)}
        
        process_log.append(f"> **蓝图摘要**：\n{plan[:200]}...\n")

        # 2. 执行
        process_log.append(f"**💻 程序员 ({roles['executor']}) 正在编码...**")
        code_res = call_agent_llm(roles["executor"], api_keys[roles["executor"]], model_config, 
                                  PROMPTS["executor"].format(plan=plan, data_context=data_context))
        
        code_match = re.search(r"```python(.*?)```", code_res, re.DOTALL)
        code = code_match.group(1).strip() if code_match else code_res
        
        process_log.append(f"**⚙️ 系统运行代码中...**")
        exec_text, exec_img, new_df = execute_callback(code, df)
        
        # 错误预检
        error_flag = False
        if exec_text.startswith("Error") or "Traceback" in exec_text:
             error_flag = True
             process_log.append(f"⚠️ **运行时错误检测到**：`{exec_text[:100]}...`")
        
        # 3. 验证
        process_log.append(f"**⚖️ 评审员 ({roles['verifier']}) 正在审核...**")
        
        # 如果有错，强制提示评审员
        force_fail_prompt = "\n\n⚠️【系统检测到运行报错】：请务必判为 FAIL 并分析报错原因！" if error_flag else ""
        
        verify_res = call_agent_llm(roles["verifier"], api_keys[roles["verifier"]], model_config, 
                                    PROMPTS["verifier"].format(plan=plan, code=code, execution_result=exec_text) + force_fail_prompt)
        
        review = extract_and_parse_json(verify_res)
        
        if not review:
            # JSON 解析失败，保险起见，如果代码没报错就通过，报错就重试
            if error_flag:
                process_log.append("❌ 评审员输出格式错误且代码报错，强制重试。")
                feedback = "代码运行报错，且评审员未返回有效 JSON。请修复代码错误。"
                iteration += 1
                continue
            else:
                process_log.append("⚠️ 评审员输出格式异常，但在无报错情况下默认通过。")
                review = {"status": "PASS", "reason": "格式解析失败但代码运行无误"}

        if review.get("status") == "PASS":
            process_log.append(f"✅ **验证通过**: {review.get('reason')}")
            return {
                "reply": f"### 🎯 Radarm 专家报告\n\n**1. 分析蓝图**\n{plan}\n\n**2. 执行结果**\n{exec_text}\n\n**3. 专家评审**\n{review.get('reason')}",
                "generated_code": code,
                "execution_result": exec_text,
                "image": exec_img,
                "new_df": new_df,
                "process_log": "\n".join(process_log)
            }
        else:
            process_log.append(f"❌ **验证不通过**: {review.get('reason')}")
            process_log.append(f"🔄 **修改建议**: {review.get('suggestion')}")
            feedback = review.get('suggestion')
            iteration += 1
            
    return {
        "reply": f"⚠️ 达到最大迭代次数 ({max_iterations})。最后一次尝试未通过验证。\n**错误信息**: {exec_text}",
        "generated_code": code,
        "execution_result": exec_text,
        "image": exec_img,
        "new_df": new_df,
        "process_log": "\n".join(process_log)
    }

# --- 核心 2: 报告生成闭环 (Report Mode) ---
def generate_expert_report(data_info, data_desc, data_sample, api_keys, model_config):
    roles = {"architect": "deepseek", "writer": "qwen", "reviewer": "zhipu"}
    
    available_keys = [k for k, v in api_keys.items() if v]
    if not available_keys: return {"error": "未配置 API Key", "log": "❌ 无 Key"}
    for r in roles: 
        if not api_keys.get(roles[r]): roles[r] = available_keys[0]

    process_log = []
    
    # 1. 架构
    process_log.append(f"### 🚀 报告生成任务启动\n")
    process_log.append(f"**🧠 学术顾问 ({roles['architect']}) 设计大纲...**")
    outline = call_agent_llm(roles["architect"], api_keys[roles["architect"]], model_config, REPORT_PROMPTS["architect"].format(data_info=data_info))
    if outline.startswith("Error"): return {"error": outline, "log": "\n".join(process_log)}
    process_log.append(f"> **大纲已生成**\n")
    
    # 2. 撰写与迭代循环
    current_feedback = ""
    iteration = 0
    max_iterations = 2
    final_content = ""
    
    while iteration < max_iterations:
        iter_prefix = f"#### [第 {iteration + 1} 轮撰写]"
        
        # 撰写
        process_log.append(f"{iter_prefix}\n**✍️ 撰稿人 ({roles['writer']}) 撰写/修改中...**")
        draft_prompt = REPORT_PROMPTS["writer"].format(
            outline=outline, 
            data_desc=data_desc, 
            data_sample=data_sample,
            feedback=current_feedback if current_feedback else "无"
        )
        draft = call_agent_llm(roles["writer"], api_keys[roles["writer"]], model_config, draft_prompt)
        if draft.startswith("Error"): return {"error": draft, "log": "\n".join(process_log)}
        
        process_log.append(f"**⚖️ 审稿人 ({roles['reviewer']}) 正在评审...**")
        review_res = call_agent_llm(roles["reviewer"], api_keys[roles["reviewer"]], model_config, REPORT_PROMPTS["reviewer"].format(draft=draft))
        
        review = extract_and_parse_json(review_res)
        
        # 如果解析失败，说明审稿人可能直接返回了润色后的文章（非 JSON 格式），这也是一种 PASS
        if not review:
            # 简单判断：如果是长文本且没有 Error，就当做是润色后的文章
            if len(review_res) > 100 and not review_res.startswith("Error"):
                process_log.append("✅ 审稿人直接返回了润色稿，流程结束。")
                final_content = review_res
                break
            else:
                process_log.append("⚠️ 审稿人返回格式异常，默认采用当前初稿。")
                final_content = draft
                break
        
        if review.get("status") == "PASS":
            process_log.append(f"✅ **审稿通过**: {review.get('comments')}")
            # 如果有润色内容就用润色内容，否则用原稿
            final_content = review.get("revised_content") if review.get("revised_content") else draft
            break
        else:
            process_log.append(f"❌ **审稿未通过**: {review.get('comments')}")
            current_feedback = review.get('comments')
            iteration += 1
            if iteration == max_iterations:
                process_log.append("⚠️ 达到最大修改次数，强制定稿。")
                final_content = draft # 没过也只能交了

    return {
        "content": final_content,
        "log": "\n".join(process_log)
    }