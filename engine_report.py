"""
数据分析报告生成引擎（五阶段，多专家混合-报告版）
================================================

本引擎只服务“生成数据分析报告”的独立工作流（/report）。

五阶段（角色固定，默认使用 DeepSeek-A/B/C 三套槽位）：
1) 数据预处理与任务拆解（项目经理：DeepSeek-C）
2) 硬核逻辑分析与计算（首席科学家：DeepSeek-A）—— 生成并执行 Python 代码，产出“中间态数据包”(JSON)
3) 业务洞察与横向关联（业务顾问：DeepSeek-B）—— 产出“洞察建议列表”(JSON)
4) 冲突解决与深度综述（首席科学家回归：DeepSeek-A）—— 产出“技术性摘要”(JSON)
5) 最终报告生成与排版（主笔：DeepSeek-C）—— 产出 Markdown 报告

输出契约（强约束）：
- Stage 1/3/4：只输出 1 个 JSON 对象，必须能被 json.loads 解析（不要 Markdown/解释文字）
- Stage 2：只输出 1 个 ```python 代码块；代码运行后必须 print 1 个 JSON（中间态数据包），尽量不要额外 print
- Stage 5：只输出 Markdown 正文
"""

import json
import re
import time
import importlib
import ast
from typing import Any, Dict, List, Optional


def _call_llm(
    provider: str,
    api_key: str,
    model_config: Dict[str, Any],
    prompt: str,
    *,
    temperature: float = 0.2,
    timeout: int = 120,
    retries: int = 2,
) -> str:
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
    payload = {"model": cfg["model"], "messages": [{"role": "user", "content": prompt}], "temperature": temperature}

    for _ in range(max(1, retries)):
        try:
            resp = requests.post(cfg["url"], headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            time.sleep(1)
    return "Error: API 调用失败"


def _extract_python_code(text: str) -> str:
    m = re.search(r"```python(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


def _extract_json_candidate(text: str) -> Optional[str]:
    """
    从模型输出/代码执行输出中尽量提取 JSON（字符串）。
    """
    if not text:
        return None

    # 1) ```json ... ```
    m = re.search(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2) 大括号对象
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return m.group(0).strip()

    # 3) 方括号数组
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        return m.group(0).strip()

    return None


def _safe_json_loads(text: str) -> Optional[Any]:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None

    # 1) strict json
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) scan first valid json object/array inside a larger text
    obj = _scan_first_json(s)
    if obj is not None:
        return obj

    # 3) normalize common JSON-ish issues then retry
    s2 = _normalize_jsonish(s)
    try:
        return json.loads(s2)
    except Exception:
        pass

    obj = _scan_first_json(s2)
    if obj is not None:
        return obj

    # 4) python-literal fallback (handles single quotes/None/True/False)
    py = _to_python_literal(s2)
    try:
        return ast.literal_eval(py)
    except Exception:
        return None


def _json_dumps(obj: Any, *, max_chars: int = 30000) -> str:
    """
    将对象序列化为 JSON 字符串，并对长度做上限保护（避免撑爆 prompt）。
    """
    s = json.dumps(obj, ensure_ascii=False, indent=2)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...(截断)"


def _call_llm_expect_json_dict(
    provider: str,
    api_key: str,
    model_config: Dict[str, Any],
    prompt: str,
    *,
    stage_name: str,
    temperature: float,
    timeout: int,
    retries: int = 2,
) -> Dict[str, Any]:
    """
    调用模型并强制拿到可解析的 JSON 对象（dict）。
    如果多次仍失败，返回 {"error": "..."}。
    """
    last_text = ""
    for attempt in range(max(1, retries)):
        last_text = _call_llm(
            provider,
            api_key,
            model_config,
            prompt,
            temperature=temperature,
            timeout=timeout,
            retries=1,
        )
        if last_text.startswith("Error"):
            return {"error": last_text}

        cand = _extract_json_candidate(last_text) or last_text.strip()
        obj = _safe_json_loads(cand)
        if obj is None:
            # 再尝试直接在完整文本中扫描 JSON（防止 regex 抓到多段 {} 导致失败）
            obj = _safe_json_loads(last_text)
        if isinstance(obj, dict):
            return {"obj": obj, "text": _json_dumps(obj)}

        # 追加一次纠错提示重试
        prompt = (
            prompt
            + "\n\n[格式纠错] 你上次输出无法被 json.loads 解析。请严格只输出一个 JSON 对象："
            + "不要 Markdown，不要代码块，不要额外解释文字，不要多余前后缀。"
        )

    return {"error": f"{stage_name} 输出无法解析为 JSON 对象", "raw": last_text[:2000]}


def _scan_first_json(text: str) -> Optional[Any]:
    """
    在包含噪声的文本中扫描第一个可解析的 JSON 对象/数组。
    """
    if not text:
        return None
    decoder = json.JSONDecoder()
    for m in re.finditer(r"[\{\[]", text):
        start = m.start()
        try:
            obj, _end = decoder.raw_decode(text[start:])
            return obj
        except Exception:
            continue
    return None


def _normalize_jsonish(s: str) -> str:
    """
    将常见的“JSON-ish”输出尽量规范化为严格 JSON（仍不保证 100%）。
    """
    t = s.strip()
    # 统一引号（中文引号/花引号）
    t = t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    # 去注释
    t = re.sub(r"//.*?$", "", t, flags=re.MULTILINE)
    t = re.sub(r"/\*[\s\S]*?\*/", "", t)
    # 去尾逗号
    t = re.sub(r",(\s*[}\]])", r"\1", t)
    # NaN/Infinity 兜底
    t = re.sub(r"\bNaN\b", "null", t)
    t = re.sub(r"\bInfinity\b", "null", t)
    t = re.sub(r"\b-Infinity\b", "null", t)
    return t


def _to_python_literal(s: str) -> str:
    """
    把 JSON-ish 字符串尽量转成 Python 字面量，供 ast.literal_eval 尝试解析。
    """
    t = s.strip()
    t = t.replace("null", "None").replace("true", "True").replace("false", "False")
    return t

def _ensure_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


DOMAIN_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "generic": {
        "kpi_definitions": [
            {
                "kpi": "记录数",
                "description": "数据表总记录数",
                "formula": "n_rows = len(df)",
                "required_columns": [],
                "group_by": [],
                "time_grain": "none",
                "directionality": "unknown",
            },
            {
                "kpi": "缺失率Top",
                "description": "缺失值占比最高的字段列表",
                "formula": "missing_pct = df.isna().mean()",
                "required_columns": [],
                "group_by": [],
                "time_grain": "none",
                "directionality": "lower_is_better",
            },
            {
                "kpi": "重复行数",
                "description": "完全重复的记录数量",
                "formula": "dup = df.duplicated().sum()",
                "required_columns": [],
                "group_by": [],
                "time_grain": "none",
                "directionality": "lower_is_better",
            },
        ],
        "trend_questions": [
            {"question": "核心数值指标是否存在明显趋势或结构性变化？", "method": "时间序列/分组对比/相关性", "required_columns": []}
        ],
        "anomaly_scan_plan": [
            {
                "name": "数值列极端值扫描",
                "method": "IQR",
                "target_columns": [],
                "threshold_or_rule": "对每个数值列使用 IQR(1.5) 找 TopN 极端值",
                "flag": "重点关注区域",
            },
            {
                "name": "逻辑不一致扫描",
                "method": "rule_based",
                "target_columns": [],
                "threshold_or_rule": "检查负值/比例>1/金额为0但数量>0等常见逻辑断层",
                "flag": "待人工复核",
            },
        ],
    },
    "sales": {
        "kpi_definitions": [
            {
                "kpi": "GMV/销售额",
                "description": "成交金额或销售总额",
                "formula": "sum(amount)",
                "required_columns": ["amount_or_revenue"],
                "group_by": [],
                "time_grain": "day",
                "directionality": "higher_is_better",
            },
            {
                "kpi": "订单数",
                "description": "订单量（去重订单ID）",
                "formula": "nunique(order_id)",
                "required_columns": ["order_id"],
                "group_by": [],
                "time_grain": "day",
                "directionality": "higher_is_better",
            },
            {
                "kpi": "客单价(AOV)",
                "description": "平均每单金额",
                "formula": "GMV / 订单数",
                "required_columns": ["amount_or_revenue", "order_id"],
                "group_by": [],
                "time_grain": "day",
                "directionality": "higher_is_better",
            },
            {
                "kpi": "用户数",
                "description": "下单/访问用户数（去重 user_id）",
                "formula": "nunique(user_id)",
                "required_columns": ["user_id"],
                "group_by": [],
                "time_grain": "day",
                "directionality": "higher_is_better",
            },
            {
                "kpi": "人均消费(ARPU)",
                "description": "GMV/用户数",
                "formula": "GMV / 用户数",
                "required_columns": ["amount_or_revenue", "user_id"],
                "group_by": [],
                "time_grain": "day",
                "directionality": "higher_is_better",
            },
        ],
        "trend_questions": [
            {"question": "GMV/销售额是否存在季节性/活动峰值（如双11）？", "method": "按天/周/月趋势 + 峰值检测", "required_columns": ["time", "amount_or_revenue"]},
            {"question": "不同渠道/地区/品类对 GMV 的贡献与趋势如何？", "method": "分组汇总 + TopN 对比", "required_columns": ["amount_or_revenue"]},
            {"question": "客单价是否发生结构性变化（上涨/下滑）？", "method": "AOV 时间序列 + 分组对比", "required_columns": ["order_id", "amount_or_revenue"]},
        ],
        "anomaly_scan_plan": [
            {
                "name": "订单金额异常",
                "method": "zscore",
                "target_columns": ["amount_or_revenue"],
                "threshold_or_rule": "|z| > 3 的单笔金额/聚合金额",
                "flag": "重点关注区域",
            },
            {
                "name": "负数/不合理值扫描（销售）",
                "method": "rule_based",
                "target_columns": ["amount_or_revenue", "quantity"],
                "threshold_or_rule": "金额<0、数量<0、金额==0但数量>0 等",
                "flag": "待人工复核",
            },
        ],
    },
    "finance": {
        "kpi_definitions": [
            {
                "kpi": "收入/入账金额",
                "description": "收入或入账总额",
                "formula": "sum(revenue)",
                "required_columns": ["revenue"],
                "group_by": [],
                "time_grain": "month",
                "directionality": "higher_is_better",
            },
            {
                "kpi": "成本/费用",
                "description": "成本或费用总额",
                "formula": "sum(cost)",
                "required_columns": ["cost"],
                "group_by": [],
                "time_grain": "month",
                "directionality": "lower_is_better",
            },
            {
                "kpi": "利润",
                "description": "利润=收入-成本（若存在利润列则直接用）",
                "formula": "revenue - cost",
                "required_columns": ["revenue", "cost"],
                "group_by": [],
                "time_grain": "month",
                "directionality": "higher_is_better",
            },
            {
                "kpi": "利润率/毛利率",
                "description": "利润率=利润/收入",
                "formula": "(revenue-cost)/revenue",
                "required_columns": ["revenue", "cost"],
                "group_by": [],
                "time_grain": "month",
                "directionality": "higher_is_better",
            },
        ],
        "trend_questions": [
            {"question": "收入/成本/利润的趋势与波动（环比/同比）如何？", "method": "时间序列 + 环比/同比", "required_columns": ["time", "revenue"]},
            {"question": "利润率异常波动是否来自成本端或收入端？", "method": "拆分贡献 + 分组对比", "required_columns": ["revenue", "cost"]},
        ],
        "anomaly_scan_plan": [
            {
                "name": "利润率异常（>100%或<-100%）",
                "method": "rule_based",
                "target_columns": ["revenue", "cost"],
                "threshold_or_rule": "利润率>1 或 < -1",
                "flag": "待人工复核",
            },
            {
                "name": "大额交易/异常波动",
                "method": "IQR",
                "target_columns": ["revenue", "cost"],
                "threshold_or_rule": "金额列 IQR 极端值 + 时间聚合突变",
                "flag": "重点关注区域",
            },
        ],
    },
    "marketing": {
        "kpi_definitions": [
            {
                "kpi": "曝光量",
                "description": "曝光总量",
                "formula": "sum(impressions)",
                "required_columns": ["impressions"],
                "group_by": [],
                "time_grain": "day",
                "directionality": "higher_is_better",
            },
            {
                "kpi": "点击量",
                "description": "点击总量",
                "formula": "sum(clicks)",
                "required_columns": ["clicks"],
                "group_by": [],
                "time_grain": "day",
                "directionality": "higher_is_better",
            },
            {
                "kpi": "CTR",
                "description": "点击率=点击/曝光",
                "formula": "clicks/impressions",
                "required_columns": ["clicks", "impressions"],
                "group_by": [],
                "time_grain": "day",
                "directionality": "higher_is_better",
            },
            {
                "kpi": "转化量",
                "description": "转化总量",
                "formula": "sum(conversions)",
                "required_columns": ["conversions"],
                "group_by": [],
                "time_grain": "day",
                "directionality": "higher_is_better",
            },
            {
                "kpi": "CVR",
                "description": "转化率=转化/点击",
                "formula": "conversions/clicks",
                "required_columns": ["conversions", "clicks"],
                "group_by": [],
                "time_grain": "day",
                "directionality": "higher_is_better",
            },
            {
                "kpi": "CPA",
                "description": "获客成本=cost/conversions",
                "formula": "cost/conversions",
                "required_columns": ["cost", "conversions"],
                "group_by": [],
                "time_grain": "day",
                "directionality": "lower_is_better",
            },
        ],
        "trend_questions": [
            {"question": "各渠道 CTR/CVR 是否稳定？是否存在明显下滑？", "method": "分渠道时间序列 + 变化点", "required_columns": ["time"]},
            {"question": "成本投放变化是否带来转化提升（ROI/边际效应）？", "method": "相关性/分段对比", "required_columns": ["cost"]},
        ],
        "anomaly_scan_plan": [
            {
                "name": "比率越界（CTR/CVR > 1）",
                "method": "rule_based",
                "target_columns": ["clicks", "impressions", "conversions"],
                "threshold_or_rule": "clicks>impressions 或 conversions>clicks",
                "flag": "待人工复核",
            }
        ],
    },
    "product": {
        "kpi_definitions": [
            {
                "kpi": "DAU",
                "description": "日活用户数",
                "formula": "nunique(user_id) by day",
                "required_columns": ["time", "user_id"],
                "group_by": [],
                "time_grain": "day",
                "directionality": "higher_is_better",
            },
            {
                "kpi": "留存率（粗略）",
                "description": "次日/7日留存（若能识别用户与时间）",
                "formula": "retention = users_returned/users_base",
                "required_columns": ["time", "user_id"],
                "group_by": [],
                "time_grain": "day",
                "directionality": "higher_is_better",
            },
        ],
        "trend_questions": [
            {"question": "DAU 是否增长？增长来自新用户还是老用户？", "method": "DAU 时间序列 + 新老用户拆分", "required_columns": ["time", "user_id"]},
        ],
        "anomaly_scan_plan": [
            {
                "name": "DAU 突变/异常峰值",
                "method": "change_point",
                "target_columns": ["time", "user_id"],
                "threshold_or_rule": "日活波动超出历史均值±3σ 或 变化点检测",
                "flag": "重点关注区域",
            }
        ],
    },
    "ops": {
        "kpi_definitions": [
            {
                "kpi": "事件/工单量",
                "description": "事件/工单数量（若存在 id 列则去重）",
                "formula": "count or nunique(ticket_id)",
                "required_columns": ["ticket_id_or_event_id"],
                "group_by": [],
                "time_grain": "day",
                "directionality": "lower_is_better",
            },
            {
                "kpi": "处理时长",
                "description": "平均处理时长/响应时长",
                "formula": "mean(duration)",
                "required_columns": ["duration"],
                "group_by": [],
                "time_grain": "day",
                "directionality": "lower_is_better",
            },
        ],
        "trend_questions": [
            {"question": "事件量是否集中在某些时间段/服务/模块？", "method": "分组 TopN + 趋势", "required_columns": ["time"]},
        ],
        "anomaly_scan_plan": [
            {
                "name": "处理时长异常（超长尾）",
                "method": "IQR",
                "target_columns": ["duration"],
                "threshold_or_rule": "IQR 极端值 + P95/P99",
                "flag": "重点关注区域",
            }
        ],
    },
}


def _normalize_sop(sop: Dict[str, Any]) -> Dict[str, Any]:
    """
    兜底修复 Stage1 SOP：保证关键字段存在，避免后续 Stage2/3/4 因缺字段崩溃。
    """
    sop = sop or {}
    sop.setdefault("data_triage", {})
    triage = sop["data_triage"] if isinstance(sop["data_triage"], dict) else {}
    sop["data_triage"] = triage
    triage.setdefault("data_shape_hint", None)
    triage.setdefault("data_form", "unknown")
    triage.setdefault("domain_guess", "unknown")
    triage.setdefault("potential_time_columns", [])
    triage.setdefault("potential_id_columns", [])
    triage.setdefault("numeric_metric_columns", [])
    triage.setdefault("categorical_dimension_columns", [])
    triage.setdefault("text_columns", [])
    triage.setdefault("known_units_or_currency", {"unit": None, "currency": None})

    sop["noise_and_quality_issues"] = _ensure_list(sop.get("noise_and_quality_issues"))
    sop["cleaning_plan"] = _ensure_list(sop.get("cleaning_plan"))
    sop.setdefault("analysis_objective", {"user_goal": "", "success_criteria": []})
    if not isinstance(sop["analysis_objective"], dict):
        sop["analysis_objective"] = {"user_goal": "", "success_criteria": []}
    sop["analysis_objective"].setdefault("user_goal", "")
    sop["analysis_objective"]["success_criteria"] = _ensure_list(sop["analysis_objective"].get("success_criteria"))

    sop["kpi_definitions"] = _ensure_list(sop.get("kpi_definitions"))
    sop["trend_questions"] = _ensure_list(sop.get("trend_questions"))
    sop["anomaly_scan_plan"] = _ensure_list(sop.get("anomaly_scan_plan"))
    sop.setdefault("confidence_scoring_rules", {"scale": "0~1", "guideline": []})
    if not isinstance(sop["confidence_scoring_rules"], dict):
        sop["confidence_scoring_rules"] = {"scale": "0~1", "guideline": []}
    sop["confidence_scoring_rules"].setdefault("scale", "0~1")
    sop["confidence_scoring_rules"]["guideline"] = _ensure_list(sop["confidence_scoring_rules"].get("guideline"))

    sop.setdefault(
        "report_requirements",
        {
            "mandatory_sections": ["执行摘要", "方法论与数据概况", "核心数据发现", "深度业务洞察", "行动建议", "风险与需人工复核", "附录"],
            "must_reference_numbers": True,
            "output_format": "markdown",
        },
    )
    if not isinstance(sop["report_requirements"], dict):
        sop["report_requirements"] = {
            "mandatory_sections": ["执行摘要", "方法论与数据概况", "核心数据发现", "深度业务洞察", "行动建议", "风险与需人工复核", "附录"],
            "must_reference_numbers": True,
            "output_format": "markdown",
        }

    sop["notes"] = _ensure_list(sop.get("notes"))
    return sop


def _apply_domain_templates(sop: Dict[str, Any]) -> Dict[str, Any]:
    """
    将行业模板“补齐”到 SOP（只在内容不足时补充，避免无谓膨胀）。
    """
    triage = sop.get("data_triage", {}) if isinstance(sop.get("data_triage"), dict) else {}
    domain = str(triage.get("domain_guess") or "unknown").strip().lower()
    tmpl = DOMAIN_TEMPLATES.get(domain, DOMAIN_TEMPLATES["generic"])

    # KPI：至少补到 6 个（generic 至少 3）
    min_kpi = 6 if domain in DOMAIN_TEMPLATES and domain != "generic" and domain != "unknown" else 3
    existing_kpis = {str(x.get("kpi")).strip() for x in sop.get("kpi_definitions", []) if isinstance(x, dict) and x.get("kpi")}
    if len(existing_kpis) < min_kpi:
        for k in tmpl.get("kpi_definitions", []):
            name = str(k.get("kpi")).strip()
            if name and name not in existing_kpis:
                sop["kpi_definitions"].append(k)
                existing_kpis.add(name)
            if len(existing_kpis) >= min_kpi:
                break

    # 趋势：至少 2 条
    if len(sop.get("trend_questions", [])) < 2:
        sop["trend_questions"].extend(tmpl.get("trend_questions", []))

    # 异常：至少 2 条
    if len(sop.get("anomaly_scan_plan", [])) < 2:
        sop["anomaly_scan_plan"].extend(tmpl.get("anomaly_scan_plan", []))

    sop["notes"] = _ensure_list(sop.get("notes"))
    sop["notes"].append(f"applied_domain_template={domain if domain in DOMAIN_TEMPLATES else 'generic'}")
    return sop


def _normalize_insights(pkg: Dict[str, Any]) -> Dict[str, Any]:
    pkg = pkg or {}
    pkg["insights"] = _ensure_list(pkg.get("insights"))
    pkg["blind_spot_checks"] = _ensure_list(pkg.get("blind_spot_checks"))
    pkg["conflicts_or_suspicions"] = _ensure_list(pkg.get("conflicts_or_suspicions"))
    pkg["questions_to_user"] = _ensure_list(pkg.get("questions_to_user"))
    pkg["assumptions"] = _ensure_list(pkg.get("assumptions"))
    pkg["notes"] = _ensure_list(pkg.get("notes"))
    return pkg


def _normalize_stage4_output(judge: Dict[str, Any], hard_pkg: Dict[str, Any], insight_pkg: Dict[str, Any]) -> Dict[str, Any]:
    judge = judge or {}
    hard_findings = _ensure_list((hard_pkg or {}).get("hard_findings"))
    hard_by_id: Dict[str, Dict[str, Any]] = {}
    for hf in hard_findings:
        if isinstance(hf, dict) and hf.get("id"):
            hard_by_id[str(hf["id"])] = hf

    judge["executive_summary_bullets"] = _ensure_list(judge.get("executive_summary_bullets"))
    judge["merged_findings"] = _ensure_list(judge.get("merged_findings"))
    judge["conflict_resolution"] = _ensure_list(judge.get("conflict_resolution"))
    judge.setdefault("technical_summary_markdown", "")
    judge["open_issues"] = _ensure_list(judge.get("open_issues"))
    judge["recommended_next_steps"] = _ensure_list(judge.get("recommended_next_steps"))
    judge["chart_plan"] = _ensure_list(judge.get("chart_plan"))

    # report_finding_rows：优先由 merged_findings 生成
    report_rows = _ensure_list(judge.get("report_finding_rows"))
    if not report_rows:
        for mf in judge["merged_findings"]:
            if not isinstance(mf, dict):
                continue
            mf_id = mf.get("id") or ""
            from_hard = _ensure_list(mf.get("from_hard"))
            review_tag = "OK"
            for hid in from_hard:
                hf = hard_by_id.get(str(hid))
                if not hf:
                    continue
                rt = str(hf.get("review_tag") or "OK")
                if rt == "待人工复核":
                    review_tag = "待人工复核"
                    break
                if rt == "重点关注区域" and review_tag == "OK":
                    review_tag = "重点关注区域"
            report_rows.append(
                {
                    "id": mf_id,
                    "final_statement": mf.get("final_statement", ""),
                    "confidence": mf.get("confidence", 0.5),
                    "review_tag": review_tag,
                    "status": mf.get("status", "TENTATIVE"),
                    "evidence_ids": from_hard,
                    "notes": mf.get("reason", ""),
                }
            )

    judge["report_finding_rows"] = report_rows

    # human_review_list：汇总硬性结论中的复核项 + 盲点审查项
    human_review = _ensure_list(judge.get("human_review_list"))
    if not human_review:
        for hf in hard_findings:
            if not isinstance(hf, dict):
                continue
            rt = str(hf.get("review_tag") or "OK")
            if rt != "OK":
                human_review.append(
                    {
                        "related_id": hf.get("id", ""),
                        "issue": hf.get("title", "需复核项"),
                        "why": hf.get("statement", ""),
                        "how_to_verify": hf.get("evidence", ""),
                    }
                )
        for bs in _ensure_list((insight_pkg or {}).get("blind_spot_checks")):
            if not isinstance(bs, dict):
                continue
            rel = _ensure_list(bs.get("related_hard_findings"))
            human_review.append(
                {
                    "related_id": rel[0] if rel else "",
                    "issue": bs.get("issue", ""),
                    "why": bs.get("why_suspicious", ""),
                    "how_to_verify": bs.get("how_to_verify", ""),
                }
            )
    judge["human_review_list"] = human_review
    return judge


PROMPTS: Dict[str, str] = {
    # Stage 1: 项目经理（智谱 GLM）
    "stage1_manager": """
你是一名【数据分析项目经理（数据分诊官）】。
你收到的是“用户原始数据（可能是 CSV/JSON/文本/日志/混合）+ 用户简要分析需求”。
你的目标不是立刻给结论，而是先做【分诊 + 清洗建议 + SOP 任务书】，让后续科学家/顾问严格按任务书执行，避免漫无目的的“泛分析”。

【输入数据（原始/混合都可能）】
{data_context}

【用户需求】
{user_request}

【你要完成的工作】
1) 数据分诊与噪声识别
- 判断数据形态：table(结构化表格)/time_series(时间序列)/text(文本评论)/log(日志)/mixed(混合)
- 列/字段识别：尽量推断可能的时间列、主键列、指标列（数值度量）、维度列（类别/地区/渠道等）、文本列
- 明确“明显格式噪声/质量风险”：如缺失/重复/编码异常/单位或币种混乱/百分号与小数混用/异常分隔符/字段含义不清/极端值等
- 给出可执行的 cleaning_plan（按步骤、说明原因、优先级）

2) 输出一份严格 JSON 的 SOP 任务书（必须能被 json.loads 解析）
SOP 的作用：让后续首席科学家可以直接按此写代码计算 KPI/趋势/异常，并产出“中间态数据包”。

【输出要求（非常重要）】
- 只输出 1 个 JSON 对象（不要 Markdown、不要多余解释文字）
- 字段必须齐全，允许值为 null/[]，但不要缺字段

【SOP JSON Schema（字段必须全部出现）】
{{
  "data_triage": {{
    "data_shape_hint": "如果能从输入推断就填写，否则写 null",
    "data_form": "table|time_series|text|log|mixed|unknown",
    "domain_guess": "sales|finance|ops|marketing|product|customer_service|other|unknown",
    "potential_time_columns": [],
    "potential_id_columns": [],
    "numeric_metric_columns": [],
    "categorical_dimension_columns": [],
    "text_columns": [],
    "known_units_or_currency": {{"unit": null, "currency": null}}
  }},
  "noise_and_quality_issues": [
    {{"issue": "...", "why_it_matters": "...", "how_to_check": "..."}}
  ],
  "cleaning_plan": [
    {{"step": "P0/P1/P2", "operation": "...(可执行描述)", "why": "...", "expected_effect": "..."}}
  ],
  "analysis_objective": {{
    "user_goal": "...(将用户需求结构化改写；若为空则写：全面EDA+业务洞察+风险扫描)",
    "success_criteria": ["...可衡量标准，如：输出Top5异常点、给出关键KPI表等"]
  }},
  "kpi_definitions": [
    {{
      "kpi": "KPI名称",
      "description": "业务含义",
      "formula": "尽量写清楚（可用自然语言或伪公式）",
      "required_columns": [],
      "group_by": [],
      "time_grain": "none|day|week|month|quarter|year",
      "directionality": "higher_is_better|lower_is_better|unknown"
    }}
  ],
  "trend_questions": [
    {{"question": "...", "method": "同比/环比/移动平均/分组对比/相关性等", "required_columns": []}}
  ],
  "anomaly_scan_plan": [
    {{
      "name": "异常扫描项名称",
      "method": "IQR|zscore|rule_based|change_point|schema_validation",
      "target_columns": [],
      "threshold_or_rule": "...(阈值/规则)",
      "flag": "待人工复核|重点关注区域|提示"
    }}
  ],
  "confidence_scoring_rules": {{
    "scale": "0~1，越高越可信",
    "guideline": [
      "样本量越大、缺失越少、定义越明确 -> 置信度越高",
      "只在局部样本/缺失严重/字段含义不清 -> 置信度降低并标记为待复核"
    ]
  }},
  "report_requirements": {{
    "mandatory_sections": ["执行摘要", "方法论与数据概况", "核心数据发现", "深度业务洞察", "行动建议", "风险与需人工复核", "附录"],
    "must_reference_numbers": true,
    "output_format": "markdown"
  }},
  "notes": []
}}
""",
    # Stage 2: 首席科学家（DeepSeek）
    "stage2_scientist_code": """
你是一名【首席数据科学家】（偏 Code/Math）。

你将收到：
- SOP 任务书（严格 JSON）
- 数据概况（可能来自 CSV/JSON/文本/混合的摘要）
- 运行环境中已存在 DataFrame `df`

你的目标不是“写报告”，而是用代码产出可复用的【中间态数据包】（Hard Findings Package）。

【SOP（JSON）】
{sop_json}

【数据概况】
{data_context}

【运行环境说明】
- 已有变量：df（Pandas DataFrame）
- 可用：pandas as pd, numpy as np（可自行 import）, matplotlib.pyplot as plt, seaborn as sns, json
- 禁止：文件读写、网络请求、os 操作、plt.show()

【列名语义匹配（非常重要）】
如果 SOP 的 required_columns 在 df.columns 中找不到，你必须先尝试“语义/关键词/模糊匹配”定位候选列，再决定能否计算：
- 时间列：date, day, time, datetime, timestamp, dt, 日期, 时间
- 金额/销售额：amount, revenue, sales, gmv, pay, 交易额, 支付金额, 销售额, 金额, 成交额
- 订单：order, order_id, 订单, trade, bill
- 用户：user, user_id, uid, customer, buyer, 用户
- 成本/费用：cost, expense, spend, 费用, 成本
- 利润：profit, margin, 利润, 毛利
- 数量：qty, quantity, count, num, 数量
- 曝光/点击：impression, pv, uv, view, 曝光, 展现；click, ctr, 点击
匹配后请把“实际命中的列名”写入 evidence/metrics，避免口径不清。

【你必须产出的中间态数据包 JSON Schema（print 输出）】
请在代码中构造一个 dict：mid = {{
  "meta": {{"generated_by": "deepseek", "assumptions": [], "limits": []}},
  "data_quality": {{
    "rows": int, "cols": int,
    "missing_top": [{{"column": str, "missing_pct": float}}],
    "duplicate_rows": int,
    "type_issues": [],
    "notes": []
  }},
  "hard_findings": [
    {{
      "id": "HF001",
      "category": "KPI|Trend|Correlation|Anomaly|LogicBreak|DataQuality",
      "title": "...",
      "statement": "...（必须包含关键数值/比例/范围）",
      "metrics": {{}},
      "confidence": 0.00,
      "evidence": "...（列/分组/时间范围/样本量n/统计口径）",
      "review_tag": "OK|待人工复核|重点关注区域",
      "severity": "low|medium|high"
    }}
  ],
  "chart_suggestions": [{{"title": "...", "chart": "line|bar|heatmap|boxplot|scatter|table", "x": "...", "y": "...", "why": "..."}}]
}}

【关键要求（非常重要）】
1) 你必须“按 SOP”尝试计算 KPI/趋势/异常；若字段缺失导致无法计算，也要生成一个 hard_finding（review_tag=待人工复核，confidence<=0.3）说明缺什么字段。
2) 每条 hard_finding 必须带 confidence（0~1），并说明证据与口径。置信度不要全给 1。
3) 异常检测必须覆盖：
   - 数值列：IQR 或 z-score 方式找极端值（只保留 TopN，例如 10 条）
   - 逻辑断层：如负销量/负人数/利润率>100% 等“疑似不合理”点，标记 review_tag=待人工复核 或 重点关注区域
4) 输出大小控制：hard_findings 最多 30 条；missing_top 最多 15；chart_suggestions 最多 8
5) 代码最后必须只 print 一份 JSON（mid），尽量不要 print 其它内容：
   print(json.dumps(mid, ensure_ascii=False, indent=2))

【输出格式】
- 只输出一个 ```python 代码块
""",
    # Stage 3: 业务顾问（DeepSeek-B）
    "stage3_consultant": """
你是一名【资深业务顾问】（擅长归因、横向知识库、风险识别）。

你将收到：
- 原始数据概况/片段（可能是表格+文本混合）
- 首席科学家计算得到的“中间态数据包”（硬性结论 JSON，含 confidence/review_tag）

你的任务：
1) 语义挖掘与归因：解释“为什么会这样”（结合业务常识/节假日/促销/季节性/渠道变化等），但必须锚定硬结论（引用 hard_findings 的 id）。
2) 多维度视角补充：从市场、用户心理、潜在风险、运营动作等维度发散，形成可执行建议。
3) 盲点审查（第一层校验）：指出硬结论中可能的业务矛盾/不可置信点（例如利润率>100%、销量为负等），给出“如何验证”的建议。

【输入：原始数据概况/片段】
{data_context}

【输入：中间态数据包（JSON）】
{hard_package_json}

【输出要求（非常重要）】
- 只输出 1 个 JSON 对象（不要 Markdown，不要解释文字）
- JSON 必须能被 json.loads 解析
- 每条洞察必须引用相关 hard_findings 的 id（based_on）

【输出 JSON Schema（字段必须全部出现）】
{{
  "insights": [
    {{
      "id": "I01",
      "title": "...",
      "why": "...（归因推理）",
      "based_on": ["HF001"],
      "confidence": 0.00,
      "actions": ["...可执行动作"],
      "risks": ["...潜在风险"]
    }}
  ],
  "blind_spot_checks": [
    {{
      "issue": "...（可能矛盾/异常的点）",
      "related_hard_findings": ["HF001"],
      "severity": "low|medium|high",
      "why_suspicious": "...",
      "how_to_verify": "...（下一步如何用数据/业务核验）"
    }}
  ],
  "conflicts_or_suspicions": [
    {{"description": "...", "related_hard_findings": ["HF001"], "suggestion": "..."}}
  ],
  "questions_to_user": [],
  "assumptions": [],
  "notes": []
}}
""",
    # Stage 4: 首席科学家回归（DeepSeek）
    "stage4_scientist_judge": """
你是一名【首席数据科学家（裁判/收敛者）】。

你将收到：
- SOP 任务书（JSON）
- 中间态数据包（硬性结论 JSON：hard_findings，含 confidence/review_tag）
- 业务顾问洞察包（JSON：insights + blind_spot_checks + conflicts）

你的任务：
1) 逻辑收敛：合并硬数据与软洞察。若冲突：
   - 默认“数据优先”，除非硬结论置信度很低或被盲点审查指出重大矛盾
   - 需要明确给出加权理由（hard vs soft 的权重）
   - 允许保留争议：标记为 DISPUTED，并列出需补充的数据/核验方式
2) 生成“技术性摘要”（用于主笔成文）：必须包含严谨逻辑链条与关键数字引用（引用 hard_findings 的 id）

【输入：SOP（JSON）】
{sop_json}

【输入：硬性结论（JSON）】
{hard_package_json}

【输入：软洞察（JSON）】
{insight_package_json}

【输出要求（非常重要）】
- 只输出 1 个 JSON 对象（不要 Markdown，不要解释文字）
- JSON 必须能被 json.loads 解析

【输出 JSON Schema（字段必须全部出现）】
{{
  "executive_summary_bullets": [],
  "merged_findings": [
    {{
      "id": "MF01",
      "from_hard": ["HF001"],
      "from_insight": ["I01"],
      "final_statement": "...（必须包含关键数值；并引用HF）",
      "confidence": 0.00,
      "status": "ACCEPTED|TENTATIVE|DISPUTED",
      "reason": "为何这样裁决（含权重逻辑）"
    }}
  ],
  "conflict_resolution": [
    {{
      "topic": "...",
      "hard_side": "...",
      "soft_side": "...",
      "decision": "...",
      "weighting": {{"hard": 0.7, "soft": 0.3}},
      "follow_up": "若仍有争议，如何核验"
    }}
  ],
  "report_finding_rows": [
    {{
      "id": "MF01",
      "final_statement": "...（报告中要展示的最终结论，包含关键数值）",
      "confidence": 0.00,
      "review_tag": "OK|待人工复核|重点关注区域",
      "status": "ACCEPTED|TENTATIVE|DISPUTED",
      "evidence_ids": ["HF001"],
      "notes": "补充口径/样本量/限制"
    }}
  ],
  "human_review_list": [
    {{
      "related_id": "HF001",
      "issue": "...（需要人工复核的点）",
      "why": "...",
      "how_to_verify": "如何核验/需要补充什么字段或业务信息"
    }}
  ],
  "technical_summary_markdown": "...（用于写报告的骨架，允许 Markdown，但放在这个字段里）",
  "open_issues": [],
  "recommended_next_steps": [],
  "chart_plan": []
}}
""",
    # Stage 5: 主笔（智谱 GLM）
    "stage5_writer": """
你是一名【专业商业分析报告主笔】（擅长结构化写作与 Markdown 排版）。

你将收到：
- SOP 任务书（JSON）
- 技术性摘要（JSON，其中 technical_summary_markdown 是报告骨架）

你的任务：
1) 把材料写成一篇完整、连贯、可交付的商业分析报告（Markdown）。
2) 严格按照结构输出：
   - 执行摘要
   - 方法论与数据概况
   - 核心数据发现（必须用表格汇总关键发现：ID/结论/置信度/review_tag）
   - 深度业务洞察
   - 行动建议（分 P0/P1/P2 优先级，尽量量化）
   - 风险与需人工复核（把 review_tag!=OK 的项列出来，并给出核验方式）
   - 附录（附上 SOP JSON 的代码块 + 术语/口径说明）

【非常重要的写作约束】
- 不要杜撰任何数字；所有数字必须来自 tech_summary_json 中的 merged_findings/report_finding_rows（并用 ID 引用）
- 如果缺少必要数字，明确写“数据不足/字段缺失”，并列出需要补充什么
- 语言客观、可审计；结论要与证据对应

【写作提示】
- “核心数据发现”的汇总表格优先使用 tech_summary_json.report_finding_rows

【输入：SOP（JSON）】
{sop_json}

【输入：技术性摘要（JSON）】
{tech_summary_json}

【输出格式】
- 只输出 Markdown 正文（不要 JSON）
""",
}


def run_report_engine(
    *,
    user_request: str,
    data_context: str,
    api_keys: Dict[str, str],
    model_config: Dict[str, Any],
    execute_callback,
    df,
) -> Dict[str, Any]:
    """
    五阶段报告生成引擎入口（供 workflow_report.py 调用）
    """
    roles = {"manager": "zhipu", "scientist": "deepseekA", "consultant": "qwen", "writer": "zhipu"}

    available_keys = [k for k, v in api_keys.items() if v]
    if not available_keys:
        return {"error": "未配置 API Key", "log": "❌ 无 Key"}
    for r in roles:
        if not api_keys.get(roles[r]):
            roles[r] = available_keys[0]

    process_log: List[str] = []
    user_req = (user_request or "").strip() or "进行全面的探索性数据分析，并输出可执行的商业分析报告"

    # --- Stage 1 ---
    process_log.append("### 🚩 阶段一：数据预处理与任务拆解（项目经理：DeepSeek-C）")
    process_log.append("正在进行数据分诊、噪声识别与 SOP 制定...")
    sop_prompt = PROMPTS["stage1_manager"].format(data_context=data_context, user_request=user_req)
    sop_result = _call_llm_expect_json_dict(
        roles["manager"],
        api_keys[roles["manager"]],
        model_config,
        sop_prompt,
        stage_name="Stage1/SOP",
        temperature=0.1,
        timeout=120,
        retries=2,
    )
    if "error" in sop_result:
        raw = sop_result.get("raw")
        if raw:
            process_log.append("\n### ⚠️ Stage1 原始输出（截断）")
            process_log.append(f"```text\n{raw}\n```")
        return {"error": sop_result["error"], "log": "\n".join(process_log)}

    sop_obj = sop_result.get("obj") if isinstance(sop_result.get("obj"), dict) else {}
    sop_obj = _normalize_sop(sop_obj)
    sop_obj = _apply_domain_templates(sop_obj)
    sop_for_next = _json_dumps(sop_obj)
    process_log.append("> **SOP（JSON）**：")
    process_log.append(f"```json\n{sop_for_next[:8000]}\n```")

    # --- Stage 2 ---
    process_log.append("\n### 🧮 阶段二：硬核逻辑分析与计算（首席科学家：DeepSeek）")
    process_log.append("正在生成 Python 分析代码（将产出中间态数据包 JSON）...")
    code_prompt = PROMPTS["stage2_scientist_code"].format(sop_json=sop_for_next, data_context=data_context)

    hard_obj: Optional[Dict[str, Any]] = None
    hard_for_next = ""
    last_exec_text = ""

    for attempt in range(2):
        code_res = _call_llm(
            roles["scientist"],
            api_keys[roles["scientist"]],
            model_config,
            code_prompt,
            temperature=0.1,
            timeout=120,
            retries=2,
        )
        if code_res.startswith("Error"):
            return {"error": code_res, "log": "\n".join(process_log)}

        code = _extract_python_code(code_res)
        process_log.append("> **Stage2 生成代码（截断预览）**：")
        process_log.append(f"```python\n{code[:2500]}\n```")
        process_log.append("⚙️ 系统正在沙盒中执行分析代码...")
        exec_text, _, _ = execute_callback(code, df)
        last_exec_text = str(exec_text)

        if isinstance(exec_text, str) and (exec_text.startswith("Error") or "Traceback" in exec_text):
            # 允许 1 次重试：把报错片段反馈给模型修复
            if attempt == 0:
                code_prompt = (
                    code_prompt
                    + f"\n\n[系统纠错] 你上次代码执行报错：{exec_text[:600]}\n"
                    + "请修复代码，并确保最后只 print 一份可被 json.loads 解析的中间态数据包 JSON（mid）。"
                )
                continue
            process_log.append(f"❌ 代码执行失败：{exec_text[:500]}")
            return {"error": "Stage2 分析代码执行失败", "log": "\n".join(process_log)}

        hard_json_text = _extract_json_candidate(str(exec_text)) or ""
        parsed = _safe_json_loads(hard_json_text) if hard_json_text else None
        if isinstance(parsed, dict):
            hard_obj = parsed
            hard_for_next = _json_dumps(hard_obj)
            break

        # 没拿到可解析 JSON，允许 1 次重试
        if attempt == 0:
            code_prompt = (
                code_prompt
                + f"\n\n[系统纠错] 你上次代码运行后没有打印出可解析 JSON。运行输出片段：{last_exec_text[:600]}\n"
                + "请修复：确保代码最后只 print(json.dumps(mid, ensure_ascii=False, indent=2))，且 mid 满足 Schema。"
            )

    if hard_obj is None:
        process_log.append(f"❌ 未拿到可解析的中间态数据包 JSON。输出片段：{last_exec_text[:600]}")
        return {"error": "Stage2 未返回有效的中间态数据包 JSON", "log": "\n".join(process_log)}

    process_log.append("> **中间态数据包（Hard Findings Package，截断预览）**：")
    process_log.append(f"```json\n{hard_for_next[:8000]}\n```")

    # --- Stage 3 ---
    process_log.append("\n### 💡 阶段三：业务洞察与横向关联（业务顾问：DeepSeek-B）")
    process_log.append("正在进行归因、发散洞察与盲点审查...")
    insight_prompt = PROMPTS["stage3_consultant"].format(data_context=data_context, hard_package_json=hard_for_next)
    insight_result = _call_llm_expect_json_dict(
        roles["consultant"],
        api_keys[roles["consultant"]],
        model_config,
        insight_prompt,
        stage_name="Stage3/洞察包",
        temperature=0.3,
        timeout=120,
        retries=2,
    )
    if "error" in insight_result:
        raw = insight_result.get("raw")
        if raw:
            process_log.append("\n### ⚠️ Stage3 原始输出（截断）")
            process_log.append(f"```text\n{raw}\n```")
        return {"error": insight_result["error"], "log": "\n".join(process_log)}

    insight_obj = insight_result.get("obj") if isinstance(insight_result.get("obj"), dict) else {}
    insight_obj = _normalize_insights(insight_obj)
    insight_for_next = _json_dumps(insight_obj)
    process_log.append("> **洞察建议列表（JSON，截断预览）**：")
    process_log.append(f"```json\n{insight_for_next[:8000]}\n```")

    # --- Stage 4 ---
    process_log.append("\n### ⚖️ 阶段四：冲突解决与深度综述（首席科学家：DeepSeek）")
    process_log.append("正在合并硬数据与软洞察，进行裁决与逻辑收敛...")
    judge_prompt = PROMPTS["stage4_scientist_judge"].format(
        sop_json=sop_for_next,
        hard_package_json=hard_for_next,
        insight_package_json=insight_for_next,
    )
    judge_result = _call_llm_expect_json_dict(
        roles["scientist"],
        api_keys[roles["scientist"]],
        model_config,
        judge_prompt,
        stage_name="Stage4/技术性摘要",
        temperature=0.1,
        timeout=120,
        retries=2,
    )
    if "error" in judge_result:
        raw = judge_result.get("raw")
        if raw:
            process_log.append("\n### ⚠️ Stage4 原始输出（截断）")
            process_log.append(f"```text\n{raw}\n```")
        return {"error": judge_result["error"], "log": "\n".join(process_log)}

    judge_obj = judge_result.get("obj") if isinstance(judge_result.get("obj"), dict) else {}
    judge_obj = _normalize_stage4_output(judge_obj, hard_obj or {}, insight_obj)
    judge_for_next = _json_dumps(judge_obj)
    process_log.append("> **技术性摘要（JSON，截断预览）**：")
    process_log.append(f"```json\n{judge_for_next[:8000]}\n```")

    # --- Stage 5 ---
    process_log.append("\n### 📝 阶段五：最终报告生成与排版（主笔：DeepSeek-C）")
    process_log.append("正在进行结构化写作与 Markdown 排版...")
    writer_prompt = PROMPTS["stage5_writer"].format(sop_json=sop_for_next, tech_summary_json=judge_for_next)
    final_report = _call_llm(roles["writer"], api_keys[roles["writer"]], model_config, writer_prompt, temperature=0.2, timeout=180)
    if final_report.startswith("Error"):
        return {"error": final_report, "log": "\n".join(process_log)}

    process_log.append("🎉 **报告生成完毕**。")
    return {"content": final_report, "log": "\n".join(process_log)}


