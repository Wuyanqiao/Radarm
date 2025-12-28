"""
多专家混合 Agent 引擎（底层能力）
------------------------------
用于“用户聊天 Radarm AI agent 的多专家混合模式”。

三角色闭环：
- Planner（架构师）：生成分析蓝图（不写代码）
- Executor（程序员）：根据蓝图写 Python 代码
- Verifier（评审）：基于执行结果判定 PASS/FAIL，并给出修复建议

支持有限轮次迭代：FAIL -> 带反馈重试。
"""

import json
import re
import importlib
from typing import Any, Dict, List, Optional

PROMPTS = {
    "planner": """
你是一名【数据分析架构师】（偏统计建模）。
运行环境说明：系统已提供 Pandas DataFrame `df`（内存数据），严禁读取任何外部文件/网络（不要 read_csv/read_excel，不要 data.csv）。

用户问题：{user_query}
数据概况：{data_context}
{feedback_context}

【重要提示 - 图片和视觉理解数据】
如果上面的"数据概况"中包含"[视觉理解]"或"[图片附件]"部分：
1. **完整理解图片信息**：仔细阅读视觉理解结果，全面理解图片中的所有信息（文字、表格、图表、标准、规范、界面元素、图像内容等）
2. **在规划中体现图片信息**：如果用户问题涉及图片中的任何信息（标准、规范、数据、文字说明、图表趋势等），**必须在规划中明确指出这些信息以及如何在分析中使用它们**
3. **充分利用图片内容**：确保规划的分析策略能够充分利用图片中的相关信息，无论是用于判定、分类、计算、验证还是其他用途
4. **信息提取策略**：如果图片包含结构化信息（表格、标准等），应在规划中明确说明如何提取和应用这些信息
5. **全面考虑**：不要只关注特定类型的信息，要全面考虑图片中的所有内容对分析任务的价值

请输出"精简但可执行"的分析蓝图（不要写代码），包含：
1) 目标变量/关键自变量（从数据概况推断可能列名，必要时列出候选映射）
2) 预处理策略（缺失、类型、异常值、派生变量如 BMI=体重/(身高^2) 的条件）
3) 建模与显著性检验（优先：相关 + 多元线性回归/广义线性；说明要输出的 p 值/置信区间/R²/样本量）
4) 诊断与下一步（共线性、残差、敏感性分析）
要求：条目化输出，最多 8 条；避免空话。
""",
    "executor": """
你是一名【建模程序员】（Python/Pandas）。
运行环境说明：系统已提供 DataFrame `df`（内存数据），严禁读取任何外部文件/网络：
- 不要 pd.read_csv/read_excel/read_*，不要使用 data.csv
- 不要 open()/os/pathlib/requests/socket/subprocess
提示：你可以使用系统预置的辅助函数 `find_col('候选1','候选2',...)` 来做列名模糊匹配（返回真实列名或 None）。

【架构师蓝图】
{plan}

【数据概况】
{data_context}

【重要提示 - 图片和视觉理解数据】
如果上面的"数据概况"中包含"[视觉理解]"或"[图片附件]"部分：
1. **完整理解图片信息**：仔细阅读视觉理解结果，理解图片中的所有信息（文字、表格、图表、标准、规范、界面元素、图像内容等）
2. **提取并使用图片信息**：根据架构师蓝图和视觉理解结果，提取图片中的任何相关信息并在代码中使用
3. **结构化数据定义**：如果图片包含表格、标准、规范、限值等结构化信息，且代码中需要使用这些信息，**必须在代码开头先解析并定义相应的数据结构**（如字典、DataFrame、列表等）
4. **示例**：
   - 如果视觉理解提到标准限值（如"总酸≥0.4（优级）"），应创建类似 `standards = {{'总酸': {{'优级': 0.4, '一级': 0.3}}}}` 的结构
   - 如果视觉理解提到表格数据，应创建相应的DataFrame或字典结构
   - 如果视觉理解提到其他结构化信息，应根据需要创建相应的数据结构
5. **避免硬编码**：确保代码中使用的图片信息都从视觉理解结果中提取并定义，而不是直接硬编码或引用未定义的变量
6. **充分利用所有信息**：不要只关注表格或标准，要充分利用图片中的任何相关信息（文字说明、图表趋势、界面状态、图像特征等）

【工具箱】
- pandas / numpy / matplotlib / seaborn
- **回归模板函数**：`fit_linear_regression(y, X, feature_names=None)` - 自动计算系数、p值、R²、置信区间（不依赖 statsmodels）
- scipy.stats：用于其他统计检验（t检验、ANOVA、相关等）
- 机器学习：ml.run(df, target='...', task='regression'/'classification'/'clustering', k=...)

【任务】
编写 Python 代码实现蓝图。要求：
1) 只能使用已存在的 df（不要加载数据）
2) 处理缺失值与类型转换：在建模前把相关列转为数值，报告有效样本量 n
3) 若用户提到 BMI 但数据没有 BMI 列：尝试用"身高/体重"推断并计算；无法推断时必须在 result 中说明缺失字段
4) 建立关系模型并检验显著性：**强烈建议使用 `fit_linear_regression(y, X, feature_names)`**，它会自动输出系数、p值、R²、置信区间等完整结果（返回 dict，可用 result = reg_result['summary'] 获取格式化文本）
5) 最终结论必须赋值给变量 result（建议是 Markdown 文本，包含结论+显著性+下一步）
6) 绘图不要 plt.show()（可选画散点+拟合线）
7) 尽量不要 print 过多内容（系统会把 print 当作最终输出）

【建议输出模板】
请将 result 组织为中文 Markdown，至少包含：
- 目标变量、主要自变量（孕周、BMI 等）、控制变量（可选）
- 模型方法（例如 OLS + 稳健标准误）
- 关键系数与 p 值（重点解释孕周/BMI）
- 样本量 n、R²/Adj.R²（或近似指标）
- 结论与下一步（若字段不足则明确缺什么）

只输出一个 ```python 代码块，不要解释。
""",
    "verifier": """
你是一名【建模评审】（严格）。

【架构师蓝图】
{plan}

【程序员代码】
{code}

【运行结果】
{execution_result}

审查红线（违反任一条必须 FAIL）：
1) 代码报错（出现 Error / Traceback / Exception）
2) 结果为空（None 或 空字符串）

请只输出标准 JSON（不要 Markdown/不要代码块/不要额外前后缀），必须能被 json.loads 解析：
{{
  "status": "PASS" 或 "FAIL",
  "reason": "通过理由或失败原因（尽量具体）",
  "suggestion": "如果 FAIL，给出可执行的修复建议（必要时指出应改哪几行/改什么）",
  "final_reply": "如果 PASS：给用户的最终答复（中文，必须基于运行结果，不要杜撰数字；若缺少关键输出就指出并给下一步）"
}}
""",
}


def _call_llm(provider: str, api_key: str, model_config: Dict[str, Any], prompt: str) -> str:
    if not api_key:
        return f"Error: 缺少 {provider} Key"
    cfg = model_config.get(provider)
    if not cfg:
        return f"Error: 未知模型 provider={provider}"

    try:
        requests = importlib.import_module("requests")
    except Exception:
        return "Error: 缺少 requests 依赖"

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {"model": cfg["model"], "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
    try:
        resp = requests.post(cfg["url"], headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            return f"Error: {resp.status_code} {resp.text}"
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"


def _extract_python_code(text: str) -> str:
    m = re.search(r"```python(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None
    # 1) strict json
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # 2) scan first json object in noisy text
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", s):
        try:
            obj, _end = dec.raw_decode(s[m.start() :])
            return obj if isinstance(obj, dict) else None
        except Exception:
            continue
    # 3) regex fallback
    try:
        m = re.search(r"\{[\s\S]*\}", s)
        if not m:
            return None
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _norm_name(s: str) -> str:
    return re.sub(r"[\s\-_]+", "", str(s or "").strip().lower())


def _pick_first(columns: List[str], candidates: List[str]) -> Optional[str]:
    cols = list(columns or [])
    norm_map = {_norm_name(c): c for c in cols}
    # 1) exact/normalized match
    for cand in candidates or []:
        cn = _norm_name(cand)
        if cn in norm_map:
            return norm_map[cn]
    # 2) substring match
    for cand in candidates or []:
        cn = _norm_name(cand)
        if not cn:
            continue
        for c in cols:
            if cn in _norm_name(c):
                return c
    return None


def _collect_contains(columns: List[str], keywords: List[str]) -> List[str]:
    out: List[str] = []
    for c in columns or []:
        n = _norm_name(c)
        if any(_norm_name(k) in n for k in keywords or []):
            out.append(c)
    # 去重保序
    seen = set()
    res = []
    for c in out:
        if c not in seen:
            res.append(c)
            seen.add(c)
    return res


def build_semantic_hints(columns: List[str]) -> Dict[str, Any]:
    """
    基于列名给出“语义候选映射”，用于帮助 LLM 更稳健地选列/建模。
    """
    cols = [str(c) for c in (columns or [])]

    core = {
        "age": _pick_first(cols, ["年龄", "age"]),
        "height": _pick_first(cols, ["身高", "height"]),
        "weight": _pick_first(cols, ["体重", "weight"]),
        "bmi": _pick_first(cols, ["孕妇BMI", "BMI", "体质指数"]),
        "gestational_week": _pick_first(cols, ["检测孕周", "孕周", "gest_week", "ga"]),
        "lmp": _pick_first(cols, ["末次月经", "LMP"]),
        "test_date": _pick_first(cols, ["检测日期", "抽血日期", "日期"]),
        "ivf": _pick_first(cols, ["IVF妊娠", "ivf"]),
        "y_conc": _pick_first(cols, ["Y染色体浓度", "胎儿Y染色体浓度", "Y浓度", "LY染色体浓度"]),
        "x_conc": _pick_first(cols, ["X染色体浓度", "X浓度"]),
        "y_z": _pick_first(cols, ["Y染色体的Z值", "Y染色体Z值", "Y Z值"]),
        "x_z": _pick_first(cols, ["X染色体的Z值", "X染色体Z值", "X Z值"]),
        "aneuploidy": _pick_first(cols, ["染色体的非整倍体", "非整倍体"]),
        "fetal_health": _pick_first(cols, ["胎儿是否健康", "是否健康"]),
    }

    chrom_z = {}
    for c in cols:
        m = re.search(r"([0-9]{1,2}|X|Y)号?染色体的Z值", c)
        if m:
            chrom_z[m.group(1)] = c

    qc_covariates = _collect_contains(
        cols,
        [
            "原始读段",
            "比对",
            "重复读段",
            "唯一比对",
            "GC含量",
            "被过滤",
        ],
    )

    # 常见分类/二元字段（可能需要 one-hot）
    categorical_candidates = _collect_contains(cols, ["是否", "妊娠", "非整倍体"])

    notes: List[str] = []
    if core.get("bmi") is None and core.get("height") and core.get("weight"):
        notes.append("BMI 列缺失：可尝试用 体重/(身高^2) 计算（注意身高单位 cm/m）。")
    if core.get("gestational_week") is None and core.get("lmp") and core.get("test_date"):
        notes.append("孕周列缺失：可尝试用 (检测日期-末次月经)/7 计算孕周（需日期可解析）。")

    return {
        "core_columns": core,
        "qc_covariates": qc_covariates,
        "chromosome_z_scores": chrom_z,
        "categorical_candidates": categorical_candidates,
        "notes": notes,
    }


def _auto_feedback_from_exec(exec_text: Any) -> str:
    t = str(exec_text or "")
    if not t:
        return ""
    if "No such file or directory" in t or "FileNotFoundError" in t:
        return "不要读取任何本地文件（如 data.csv）。运行环境已提供 df，请直接使用 df 进行分析。"
    if "KeyError" in t:
        return "出现 KeyError（列名不存在）。请用 df.columns 检查真实列名，做模糊匹配/候选映射，并在建模前统一重命名。"
    if "ModuleNotFoundError" in t:
        return "环境缺少某些第三方库。请避免使用不可用库，或对 import 做 try/except 并提供 pandas/numpy 的替代实现。"
    if "禁止文件/网络/系统操作" in t:
        return "系统禁止文件/网络/系统操作。请移除 read_csv/read_excel/open/requests 等，直接使用 df 进行计算。"
    return ""


def _provider_label(provider: str) -> str:
    """
    仅用于 UI/日志展示：把内部 provider id 映射为用户可理解的 DeepSeek 槽位名。
    """
    mapping = {
        "deepseekA": "DeepSeek-A",
        "deepseekB": "DeepSeek-B",
        "deepseekC": "DeepSeek-C",
        "zhipu": "Zhipu",
        "qwen": "Qwen",
    }
    return mapping.get(str(provider), str(provider))


def run_multi_agent_engine(
    *,
    user_query: str,
    data_context: str,
    api_keys: Dict[str, str],
    model_config: Dict[str, Any],
    roles: Optional[Dict[str, str]] = None,
    execute_callback,
    df,
) -> Dict[str, Any]:
    """
    多专家混合 Agent 引擎入口（供 workflow_multi_chat.py 调用）
    """
    roles = roles or {"planner": "deepseekA", "executor": "deepseekB", "verifier": "deepseekC"}

    # Key 检查与自动补位（允许只配 1 个 key，但效果会下降）
    available_keys = [k for k, v in api_keys.items() if v]
    if not available_keys:
        return {"error": "未配置 API Key", "process_log": "❌ 无 Key"}
    for r in roles:
        if not api_keys.get(roles[r]):
            roles[r] = available_keys[0]

    process_log: List[str] = []
    iteration = 0
    max_iterations = 2
    feedback = ""

    while iteration < max_iterations:
        iter_prefix = f"#### [第 {iteration + 1} 轮迭代]"
        feedback_context = f"\n\n[上一轮反馈]\n{feedback}\n" if feedback else ""

        # 给模型提供“列语义候选映射”，提升选列与建模命中率
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

        # 1) 规划
        process_log.append(f"{iter_prefix}\n**🧠 架构师 ({_provider_label(roles['planner'])}) 正在规划...**")
        plan = _call_llm(
            roles["planner"],
            api_keys[roles["planner"]],
            model_config,
            PROMPTS["planner"].format(user_query=user_query, data_context=enriched_data_context, feedback_context=feedback_context),
        )
        if plan.startswith("Error"):
            return {"error": plan, "process_log": "\n".join(process_log)}
        process_log.append(f"> **蓝图摘要**：\n{plan[:200]}...\n")

        # 2) 执行
        process_log.append(f"**💻 程序员 ({_provider_label(roles['executor'])}) 正在编码...**")
        code_res = _call_llm(
            roles["executor"],
            api_keys[roles["executor"]],
            model_config,
            PROMPTS["executor"].format(plan=plan, data_context=enriched_data_context),
        )
        code = _extract_python_code(code_res)

        process_log.append("**⚙️ 运行代码...**")
        # 支持新的4元组返回： (output_text, image_path, plotly_json, new_df)
        result = execute_callback(code, df)
        if len(result) == 4:
            exec_text, exec_img, plotly_json, new_df = result
        else:
            # 向后兼容：如果是3元组，添加 None 作为 plotly_json
            exec_text, exec_img, new_df = result
            plotly_json = None

        has_error = False
        if isinstance(exec_text, str) and (exec_text.startswith("Error") or "Traceback" in exec_text):
            has_error = True
            process_log.append(f"⚠️ **报错**: `{exec_text[:120]}...`")

        # 已知报错：系统自动反馈（更快，不必等评审员）
        auto_fb = _auto_feedback_from_exec(exec_text)
        if has_error and auto_fb:
            process_log.append(f"🤖 **系统自动诊断**: {auto_fb}")
            feedback = auto_fb
            iteration += 1
            continue

        # 3) 验证
        process_log.append(f"**⚖️ 评审员 ({_provider_label(roles['verifier'])}) 正在审核...**")
        force_fail = "\n\n⚠️ 代码报错，请判 FAIL 并说明原因与修复建议！" if has_error else ""
        verify_res = _call_llm(
            roles["verifier"],
            api_keys[roles["verifier"]],
            model_config,
            PROMPTS["verifier"].format(plan=plan, code=code, execution_result=exec_text) + force_fail,
        )

        review = _extract_json(verify_res)
        if not isinstance(review, dict):
            review = {}
        status = str(review.get("status") or "").upper()
        reason = str(review.get("reason") or "").strip() or "评审未给出明确原因"
        suggestion = str(review.get("suggestion") or "").strip()
        final_reply = str(review.get("final_reply") or "").strip()

        # 评审输出不可解析/缺字段时：不要默认 PASS
        if status not in ("PASS", "FAIL"):
            status = "FAIL" if has_error else "FAIL"
            if not suggestion:
                suggestion = "评审输出无法解析为合法 JSON。请严格按 JSON Schema 输出，并修复代码/结果为空等问题。"

        if status == "PASS":
            process_log.append(f"✅ **验证通过**: {reason}")
            return {
                "reply": final_reply
                if final_reply
                else f"### 🎯 Radarm 多专家结论\n\n**结论**: {exec_text}\n\n**评审**: {reason}",
                "generated_code": code,
                "execution_result": exec_text,
                "image": exec_img,
                "plotly_json": plotly_json,  # 新增：Plotly 图表 JSON
                "new_df": new_df,
                "process_log": "\n".join(process_log),
            }

        process_log.append(f"❌ **驳回**: {reason}\n🔄 **建议**: {suggestion}")
        feedback = suggestion
        iteration += 1

    return {
        "reply": f"⚠️ 达到最大迭代次数。最后结果: {exec_text}",
        "generated_code": code,
        "execution_result": exec_text,
        "image": exec_img,
        "plotly_json": plotly_json,  # 新增：Plotly 图表 JSON
        "new_df": new_df,
        "process_log": "\n".join(process_log),
    }


