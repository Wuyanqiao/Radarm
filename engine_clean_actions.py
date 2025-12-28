from __future__ import annotations

import json
import re
import time
import importlib
from typing import Any, Dict, List, Optional

from action_engine import SUPPORTED_ACTION_TYPES, resolve_columns


def _call_llm(
    provider: str,
    api_key: str,
    model_config: Dict[str, Any],
    prompt: str,
    *,
    temperature: float = 0.1,
    timeout: int = 90,
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

    last_err = ""
    for _ in range(max(1, retries)):
        try:
            resp = requests.post(cfg["url"], headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            last_err = f"{resp.status_code} {resp.text}"
        except Exception as e:
            last_err = str(e)
            time.sleep(0.5)
    return f"Error: API 调用失败 {last_err}"


def _extract_json_candidate(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return m.group(0).strip()
    return None


def _safe_json_loads(text: str) -> Optional[Any]:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    # 扫描第一段 json 对象
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", s):
        try:
            obj, _end = dec.raw_decode(s[m.start() :])
            return obj
        except Exception:
            continue
    return None


def _choose_provider(api_keys: Dict[str, str], primary_model: str) -> Optional[str]:
    if api_keys.get(primary_model):
        return primary_model
    for k, v in api_keys.items():
        if v:
            return k
    return None


PROMPT = """
你是一个【数据清洗/变换助手】。你的任务是把用户的自然语言需求转换成"可执行的动作(Action)列表"，用于对 DataFrame 进行清洗/变换。

【用户需求】
{user_query}

【数据概况】
{data_context}

【重要提示 - 图片和视觉理解数据】
如果上面的"数据概况"中包含"[视觉理解]"或"[图片附件]"部分：
1. **完整理解图片信息**：仔细阅读视觉理解结果，全面理解图片中的所有信息（文字、表格、图表、标准、规范、界面元素、图像内容等）
2. **在建议中体现图片信息**：如果用户需求涉及图片中的任何信息（标准、规范、数据要求、文字说明等），请在清洗建议中体现对这些信息的理解和应用
3. **确保建议符合图片要求**：确保清洗建议符合图片中提到的任何标准、要求或规范
4. **充分利用所有信息**：不要只关注特定类型的信息，要全面考虑图片中的所有内容对清洗任务的价值

【允许的 Action 类型（只能从这里选）】
- replace_missing: 将指定值设为缺失值(NaN)
  params: {{ "columns": ["列1","列2"] 或 [], "values": [999,"拒绝回答"] }}
- dropna_rows: 删除缺失值行
  params: {{ "columns": ["列1"] 或 [], "how": "any|all" }}
- fillna: 填充缺失值
  params: {{ "columns": ["列1"] 或 [], "strategy": "value|mean|median|mode|zero", "value": 任意(仅 strategy=value 时需要) }}
- cast_type: 类型转换
  params: {{ "column": "列名", "to": "float|int|string|category|datetime" }}
- standardize: 标准化
  params: {{ "columns": ["列1","列2"], "method": "zscore" }}
- winsorize: 缩尾/截尾（按分位数裁剪异常值）
  params: {{ "columns": ["列1","列2"], "lower": 0.01, "upper": 0.99 }}
- one_hot_encode: 虚拟变量转换(one-hot)
  params: {{ "columns": ["列1","列2"], "drop_first": false, "prefix_sep": "=" }}
- rename_columns: 重命名列
  params: {{ "mapping": {{ "旧列名": "新列名" }} }}
- drop_columns: 删除列
  params: {{ "columns": ["列1","列2"] }}
- deduplicate: 去重
  params: {{ "subset": ["列1","列2"] 或 [], "keep": "first|last" }}
- trim_whitespace: 去除字符串首尾空格
  params: {{ "columns": ["列1","列2"] 或 [] }}

【输出要求（非常重要）】
1) 只输出 1 个 JSON 对象（不要 Markdown/不要解释文字/不要代码块）
2) JSON 必须能被 json.loads 解析
3) actions 最多 6 个；如果用户表述不清，可让 actions 为空，并在 reply_to_user 里提出需要确认的问题
4) columns/column 必须来自数据概况中的真实列名（不要编造）

【输出 JSON Schema（字段必须全部出现）】
{{
  "reply_to_user": "...（用中文，告诉用户你打算做哪些清洗动作，并提示需要确认的风险）",
  "actions": [
    {{ "type": "replace_missing", "params": {{}} }}
  ],
  "risk_notes": ["...（可为空数组）"]
}}
""".strip()


def suggest_clean_actions(
    *,
    user_query: str,
    data_context: str,
    api_keys: Dict[str, str],
    primary_model: str,
    model_config: Dict[str, Any],
    available_columns: List[str],
) -> Dict[str, Any]:
    provider = _choose_provider(api_keys, primary_model)
    if not provider:
        return rule_based_suggest(user_query=user_query, available_columns=available_columns)

    prompt = PROMPT.format(user_query=user_query, data_context=data_context)
    text = _call_llm(provider, api_keys.get(provider, ""), model_config, prompt, temperature=0.1, timeout=90, retries=2)
    if text.startswith("Error"):
        return rule_based_suggest(user_query=user_query, available_columns=available_columns, error=text)

    cand = _extract_json_candidate(text) or text
    obj = _safe_json_loads(cand)
    if not isinstance(obj, dict):
        return rule_based_suggest(user_query=user_query, available_columns=available_columns, error="AI 输出无法解析为 JSON")

    reply = str(obj.get("reply_to_user") or "").strip() or "我给出了一组清洗/变换建议，请确认后应用。"
    actions = obj.get("actions") or []
    risk_notes = obj.get("risk_notes") or []

    if not isinstance(actions, list):
        actions = []
    if not isinstance(risk_notes, list):
        risk_notes = [str(risk_notes)]

    normalized_actions: List[Dict[str, Any]] = []
    for a in actions[:6]:
        if not isinstance(a, dict):
            continue
        t = a.get("type")
        if t not in SUPPORTED_ACTION_TYPES:
            continue
        params = a.get("params") if isinstance(a.get("params"), dict) else {}

        # 列名解析
        if t in ("replace_missing", "dropna_rows", "fillna", "standardize", "winsorize", "one_hot_encode", "drop_columns", "trim_whitespace"):
            cols = params.get("columns") or []
            if isinstance(cols, str):
                cols = [cols]
            if isinstance(cols, list) and cols:
                params["columns"] = resolve_columns(cols, available_columns)
        if t == "cast_type":
            col = params.get("column")
            if isinstance(col, str) and col:
                resolved = resolve_columns([col], available_columns)
                if resolved:
                    params["column"] = resolved[0]
        if t == "rename_columns":
            mapping = params.get("mapping")
            if isinstance(mapping, dict):
                new_map = {}
                for k, v in mapping.items():
                    rk = resolve_columns([str(k)], available_columns)
                    if rk and v:
                        new_map[rk[0]] = str(v)
                params["mapping"] = new_map
        if t == "deduplicate":
            subset = params.get("subset") or []
            if isinstance(subset, str):
                subset = [subset]
            if isinstance(subset, list) and subset:
                params["subset"] = resolve_columns(subset, available_columns)

        normalized_actions.append({"type": t, "params": params})

    # 如果模型未给出任何动作，则给出安全兜底：删除含缺失行 + 将 999 设为缺失；若能找到“score/得分/分数”列则再做标准化
    if not normalized_actions:
        fallback_actions: List[Dict[str, Any]] = [
            {"type": "dropna_rows", "params": {"columns": [], "how": "any"}},
            {"type": "replace_missing", "params": {"columns": [], "values": [999, "999"]}},
        ]
        score_cols = resolve_columns(["score", "得分", "分数"], available_columns)
        if score_cols:
            fallback_actions.append({"type": "standardize", "params": {"columns": score_cols, "method": "zscore"}})
        normalized_actions = fallback_actions
        if not reply:
            reply = "我未从模型获得具体动作，已提供默认清洗：删除含缺失行、将 999 设为缺失" + ("，并标准化 score/得分列" if score_cols else "")

    return {"reply": reply, "suggested_actions": normalized_actions, "risk_notes": risk_notes, "provider": provider}


def rule_based_suggest(*, user_query: str, available_columns: List[str], error: str = "") -> Dict[str, Any]:
    q = (user_query or "").strip()
    actions: List[Dict[str, Any]] = []
    risk: List[str] = []

    # 999 -> 缺失
    if re.search(r"\b999\b", q) or "999" in q:
        actions.append({"type": "replace_missing", "params": {"columns": [], "values": [999, "999"]}})
        risk.append("请确认 999 是否为业务上的合法值；若是，则不应替换为缺失值。")

    # 删除空值行
    if "删除空值" in q or "删除缺失" in q or "去掉空值" in q:
        actions.append({"type": "dropna_rows", "params": {"columns": [], "how": "any"}})

    # 去重
    if "去重" in q or "删除重复" in q:
        actions.append({"type": "deduplicate", "params": {"subset": [], "keep": "first"}})

    # 缩尾/截尾
    if "缩尾" in q or "截尾" in q or "winsor" in q.lower():
        actions.append({"type": "winsorize", "params": {"columns": [], "lower": 0.01, "upper": 0.99}})
        risk.append("缩尾会裁剪极端值，请确认分位数范围（默认 1%~99%）。")

    # one-hot / 虚拟变量
    if "onehot" in q.lower() or "one-hot" in q.lower() or "虚拟变量" in q or "哑变量" in q:
        actions.append({"type": "one_hot_encode", "params": {"columns": [], "drop_first": False, "prefix_sep": "="}})
        risk.append("one-hot 会增加列数；若类别很多可能导致维度膨胀。")

    # 标准化
    m = re.search(r"标准化\s*([^\s，,。]+)", q)
    if m:
        target = m.group(1)
        cols = resolve_columns([target], available_columns)
        if cols:
            actions.append({"type": "standardize", "params": {"columns": cols, "method": "zscore"}})

    reply = "我给出了一组清洗/变换建议，请确认后应用。"
    if error:
        reply = f"（提示：AI 建议生成失败，已使用规则兜底）\n{reply}"

    return {"reply": reply, "suggested_actions": actions[:6], "risk_notes": risk, "provider": "rule_based"}


