"""
Report Workflow v2 (multi-model, artifact-rich)
==============================================

目标：
- 允许在一键生成报告时，灵活使用 DeepSeek(chat/reasoner)、Zhipu(GLM-4.5/4.6/4.7)、Qwen(qwen-max/long/coder) 等模型参与不同阶段
- 生成报告的同时产出：图表（PNG）+ 图表数据（CSV）+ manifest（JSON）
- 报告产物保存到 out/{session_id}/reports/{report_id}/ 下，便于导出与多版本管理

约束：
- 不在此处做 FastAPI 路由；由 backend.py 调用
"""

from __future__ import annotations

import json
import re
import time
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from analysis_engine import run_analysis, TableOut, ChartOut, _safe_rel_path  # type: ignore


def _call_llm(
    provider: str,
    api_key: str,
    model_config: Dict[str, Any],
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    timeout: int = 180,
) -> str:
    if not api_key:
        return f"Error: 缺少 {provider} Key"
    cfg = model_config.get(provider)
    if not cfg:
        return f"Error: 未知 provider={provider}"
    try:
        requests = importlib.import_module("requests")
    except Exception:
        return "Error: 缺少 requests 依赖"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": (model or cfg.get("model")),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
    }
    try:
        resp = requests.post(cfg.get("url"), headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 200:
            return f"Error: {resp.status_code} {resp.text}"
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"


def _scan_first_json(text: str) -> Optional[Any]:
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None
    dec = json.JSONDecoder()
    for m in re.finditer(r"[\{\[]", s):
        try:
            obj, _end = dec.raw_decode(s[m.start() :])
            return obj
        except Exception:
            continue
    return None


def _extract_json_candidate(text: str) -> str:
    if not text:
        return ""
    s = str(text)
    m = re.search(r"```json(.*?)```", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        return m.group(0).strip()
    m = re.search(r"\[[\s\S]*\]", s)
    if m:
        return m.group(0).strip()
    return s.strip()


def _safe_json_loads(text: str) -> Optional[Any]:
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    obj = _scan_first_json(s)
    return obj


def _truncate(s: str, n: int) -> str:
    s2 = str(s or "")
    return s2 if len(s2) <= n else s2[:n] + "\n...(截断)"


def _stage_pick(stage_models: Dict[str, Any], stage: str, default_provider: str, default_model: str) -> Tuple[str, str]:
    cfg = (stage_models or {}).get(stage) if isinstance((stage_models or {}).get(stage), dict) else {}
    provider = str(cfg.get("provider") or default_provider)
    model = str(cfg.get("model") or default_model)
    return provider, model


def _ensure_report_dir(session_id: str, report_id: str) -> Tuple[Path, str]:
    sid = _safe_rel_path(session_id)
    rid = _safe_rel_path(report_id)
    rel = f"{sid}/reports/{rid}"
    p = Path("out") / sid / "reports" / rid
    p.mkdir(parents=True, exist_ok=True)
    (p / "data").mkdir(parents=True, exist_ok=True)
    return p, rel


def _save_tables_csv(report_dir: Path, tables: List[TableOut], *, job_key: str) -> List[Dict[str, Any]]:
    out = []
    for idx, t in enumerate(tables or []):
        name = getattr(t, "name", f"表格{idx+1}")
        md = getattr(t, "markdown", "") or ""
        df = getattr(t, "df", None)
        csv_rel = None
        if isinstance(df, pd.DataFrame) and len(df.columns) > 0:
            fname = f"data/{job_key}_table_{idx+1}.csv"
            try:
                df.to_csv(report_dir / fname, index=False, encoding="utf-8-sig")
                csv_rel = fname
            except Exception:
                csv_rel = None
        out.append({"name": name, "markdown": md, "csv": csv_rel})
    return out


def run_report_engine_v2(
    *,
    session_id: str,
    report_id: str,
    user_request: str,
    data_context: str,
    api_keys: Dict[str, str],
    model_config: Dict[str, Any],
    df: pd.DataFrame,
    stage_models: Dict[str, Any] = None,
    selected_columns: Optional[List[str]] = None,
    sample_rows: Optional[int] = None,
    check_cancelled: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    返回：
    {
      report_id, title, content, log,
      artifacts: { base_dir, charts:[...], tables:[...], manifest_path, report_path },
      plan: {...}, insights: {...}
    }
    """
    stage_models = stage_models or {}
    t0 = time.time()
    log: List[str] = []

    report_dir, base_rel = _ensure_report_dir(session_id, report_id)

    # subset info (仅用于写入 manifest/提示，不在此处筛 df；筛选由 backend 完成后传入 df)
    selected_columns = selected_columns or []
    sample_rows = int(sample_rows) if sample_rows else None

    # Stage A: plan
    planner_provider, planner_model = _stage_pick(stage_models, "planner", "deepseekA", "deepseek-reasoner")
    log.append(f"### 🧭 Stage A：规划（{planner_provider} / {planner_model}）")
    allowed = [
        "overview",
        "descriptive",
        "frequency",
        "crosstab",
        "group_summary",
        "normality",
        "correlation",
        "linear_regression",
        "logistic_regression",
        "pca",
        "kmeans",
        "ttest",
        "anova",
        "chi_square",
        "nonparam",
    ]
    plan_prompt = (
        "你是 Radarm 的【报告规划器】。请基于用户需求与数据概况，规划一份可交付的数据分析报告。\n"
        "你必须输出 1 个严格 JSON 对象（不要 Markdown/不要代码块/不要前后缀）。\n"
        "JSON Schema：\n"
        "{\n"
        '  "title": "报告标题",\n'
        '  "jobs": [ {"analysis": "...", "params": {...}}, ... ],\n'
        '  "sections": [ {"title": "...", "job_indexes": [0,1,...], "notes": "..."}, ... ],\n'
        '  "assumptions": ["..."],\n'
        '  "risks": ["..."]\n'
        "}\n"
        "约束：\n"
        f"- analysis 只能从以下列表选择：{allowed}\n"
        "- jobs 最多 10 个\n"
        "- params 中涉及列名必须使用真实列名（来自数据概况）\n\n"
        f"[用户需求]\n{user_request}\n\n"
        f"[数据概况]\n{_truncate(data_context, 12000)}\n"
    )
    plan_text = _call_llm(planner_provider, api_keys.get(planner_provider, ""), model_config, plan_prompt, model=planner_model, temperature=0.2, timeout=180)
    if plan_text.startswith("Error"):
        return {"error": plan_text, "log": "\n".join(log)}
    plan_obj = _safe_json_loads(_extract_json_candidate(plan_text))
    if not isinstance(plan_obj, dict):
        # fallback plan
        plan_obj = {
            "title": "数据分析报告",
            "jobs": [{"analysis": "overview", "params": {}}, {"analysis": "descriptive", "params": {"columns": []}}],
            "sections": [{"title": "数据概览与描述统计", "job_indexes": [0, 1], "notes": "自动兜底：概览+描述统计"}],
            "assumptions": [],
            "risks": ["规划输出无法解析，已启用兜底方案"],
        }
    # sanitize jobs
    jobs_in = plan_obj.get("jobs") or []
    jobs: List[Dict[str, Any]] = []
    for j in jobs_in[:10]:
        if not isinstance(j, dict):
            continue
        a = str(j.get("analysis") or "").strip()
        if a not in allowed:
            continue
        p = j.get("params") or {}
        jobs.append({"analysis": a, "params": p if isinstance(p, dict) else {}})
    if not jobs:
        jobs = [{"analysis": "overview", "params": {}}, {"analysis": "descriptive", "params": {"columns": []}}]
    plan_obj["jobs"] = jobs

    # 检查是否已取消
    if check_cancelled and check_cancelled():
        log.append("\n⚠️ 报告生成已取消（规划阶段后）")
        return {"cancelled": True, "log": "\n".join(log), "error": "报告生成已取消"}

    # Stage B: execute deterministic analyses (charts+tables)
    log.append("\n### 🧮 Stage B：计算与产物（确定性引擎）")
    artifacts_tables: List[Dict[str, Any]] = []
    artifacts_charts: List[Dict[str, Any]] = []
    job_results: List[Dict[str, Any]] = []

    out_subdir = f"reports/{_safe_rel_path(report_id)}"
    for idx, job in enumerate(jobs):
        # 检查是否已取消
        if check_cancelled and check_cancelled():
            log.append(f"\n⚠️ 报告生成已取消（执行到第 {idx+1}/{len(jobs)} 个分析）")
            return {"cancelled": True, "log": "\n".join(log), "error": "报告生成已取消"}
        analysis = job.get("analysis")
        params = job.get("params") or {}
        job_key = f"job{idx+1}_{analysis}"
        log.append(f"- 运行：{analysis}  params={_truncate(json.dumps(params, ensure_ascii=False), 400)}")
        try:
            res = run_analysis(session_id=session_id, df=df, analysis=str(analysis), params=params, out_subdir=out_subdir)
        except Exception as e:
            log.append(f"  ⚠️ 失败：{str(e)}")
            continue

        tables: List[TableOut] = res.get("tables") or []
        charts: List[ChartOut] = res.get("charts") or []
        summary = res.get("summary") or {}

        t_items = _save_tables_csv(report_dir, tables, job_key=job_key)
        for t in t_items:
            # 把 csv 相对路径补全为 /out 相对路径
            csv_rel = t.get("csv")
            if csv_rel:
                t["csv_path"] = f"{base_rel}/{csv_rel}"
            else:
                t["csv_path"] = None
            artifacts_tables.append({"job": job_key, **t})

        for c in charts or []:
            name = getattr(c, "name", "图")
            path = getattr(c, "path", None)
            if path:
                artifacts_charts.append({"job": job_key, "name": name, "path": path})

        job_results.append(
            {
                "job": job_key,
                "analysis": analysis,
                "params": params,
                "tables": [{"name": t.get("name"), "csv_path": t.get("csv_path"), "markdown": _truncate(t.get("markdown") or "", 3000)} for t in t_items],
                "charts": [{"name": x.get("name"), "path": x.get("path")} for x in artifacts_charts if x.get("job") == job_key],
                "summary": summary,
            }
        )

    # 检查是否已取消
    if check_cancelled and check_cancelled():
        log.append("\n⚠️ 报告生成已取消（计算阶段后）")
        return {"cancelled": True, "log": "\n".join(log), "error": "报告生成已取消"}

    # Stage C: insights
    analyst_provider, analyst_model = _stage_pick(stage_models, "analyst", "deepseekB", "deepseek-reasoner")
    log.append(f"\n### 💡 Stage C：洞察（{analyst_provider} / {analyst_model}）")
    insight_prompt = (
        "你是一名严谨的【业务洞察分析师】。请基于下述分析结果，提炼洞察（不要杜撰数字）。\n"
        "只输出 1 个严格 JSON 对象：\n"
        "{\n"
        '  "findings": [ {"id":"F1","title":"...","evidence":"引用表格/统计量","confidence":"high|mid|low","tags":["..."]}, ... ],\n'
        '  "next_steps": ["..."],\n'
        '  "data_issues": ["..."]\n'
        "}\n\n"
        f"[用户需求]\n{user_request}\n\n"
        f"[分析结果(截断)]\n{_truncate(json.dumps(job_results, ensure_ascii=False, indent=2), 14000)}\n"
    )
    insight_text = _call_llm(analyst_provider, api_keys.get(analyst_provider, ""), model_config, insight_prompt, model=analyst_model, temperature=0.3, timeout=180)
    insights_obj = _safe_json_loads(_extract_json_candidate(insight_text))
    if not isinstance(insights_obj, dict):
        insights_obj = {"findings": [], "next_steps": [], "data_issues": ["洞察阶段输出无法解析，已跳过。"]}

    # 检查是否已取消
    if check_cancelled and check_cancelled():
        log.append("\n⚠️ 报告生成已取消（洞察阶段后）")
        return {"cancelled": True, "log": "\n".join(log), "error": "报告生成已取消"}

    # Stage D: write markdown
    writer_provider, writer_model = _stage_pick(stage_models, "writer", "zhipu", "glm-4.7")
    log.append(f"\n### 📝 Stage D：成文（{writer_provider} / {writer_model}）")

    charts_index = "\n".join([f"- {c['name']}: /out/{c['path']}" for c in artifacts_charts[:30]])
    tables_index = "\n".join([f"- {t['name']}: {t.get('csv_path') or '（无CSV）'}" for t in artifacts_tables[:30]])

    writer_prompt = (
        "你是一名【数据分析报告主笔】。请把材料写成一份可交付报告（Markdown）。\n"
        "强约束：\n"
        "- 不要杜撰任何数字；涉及统计量/数值必须来自给定的 job_results/summary 或表格。\n"
        "- 允许写清楚“数据不足/字段缺失”。\n"
        "- 报告需包含：执行摘要、数据概况、核心发现（分点+证据）、图表解读、建议与风险、附录。\n"
        "- 图表可用 Markdown 图片语法引用：![](/out/<path>)。\n\n"
        f"[报告标题(建议)]\n{plan_obj.get('title','数据分析报告')}\n\n"
        f"[用户需求]\n{user_request}\n\n"
        f"[报告结构建议]\n{_truncate(json.dumps(plan_obj.get('sections') or [], ensure_ascii=False, indent=2), 4000)}\n\n"
        f"[洞察(JSON)]\n{_truncate(json.dumps(insights_obj, ensure_ascii=False, indent=2), 9000)}\n\n"
        f"[分析结果(截断 JSON)]\n{_truncate(json.dumps(job_results, ensure_ascii=False, indent=2), 14000)}\n\n"
        f"[图表索引]\n{_truncate(charts_index, 3000)}\n\n"
        f"[表格数据索引(CSV)]\n{_truncate(tables_index, 3000)}\n\n"
        "请只输出 Markdown 正文。\n"
    )
    draft_md = _call_llm(writer_provider, api_keys.get(writer_provider, ""), model_config, writer_prompt, model=writer_model, temperature=0.2, timeout=240)
    if draft_md.startswith("Error"):
        return {"error": draft_md, "log": "\n".join(log)}

    # 检查是否已取消
    if check_cancelled and check_cancelled():
        log.append("\n⚠️ 报告生成已取消（成文阶段后）")
        return {"cancelled": True, "log": "\n".join(log), "error": "报告生成已取消"}

    # Stage E: reviewer (optional)
    reviewer_provider, reviewer_model = _stage_pick(stage_models, "reviewer", "deepseekC", "deepseek-reasoner")
    log.append(f"\n### 🧪 Stage E：审校（{reviewer_provider} / {reviewer_model}）")
    review_prompt = (
        "你是一名【严格审校员】。任务：检查报告是否存在杜撰数字、证据不足、逻辑跳跃。\n"
        "只输出 1 个严格 JSON 对象：\n"
        "{\n"
        '  "status": "PASS"|"FAIL",\n'
        '  "issues": ["..."],\n'
        '  "fixed_markdown": "如果 FAIL，给出修订后的完整 Markdown；如果 PASS 可留空"\n'
        "}\n\n"
        f"[分析结果(截断)]\n{_truncate(json.dumps(job_results, ensure_ascii=False, indent=2), 12000)}\n\n"
        f"[报告草稿]\n{_truncate(draft_md, 12000)}\n"
    )
    review_text = _call_llm(reviewer_provider, api_keys.get(reviewer_provider, ""), model_config, review_prompt, model=reviewer_model, temperature=0.1, timeout=200)
    review_obj = _safe_json_loads(_extract_json_candidate(review_text))
    final_md = draft_md
    if isinstance(review_obj, dict) and str(review_obj.get("status")).upper() == "FAIL":
        fixed = str(review_obj.get("fixed_markdown") or "").strip()
        if fixed:
            final_md = fixed
        issues = review_obj.get("issues") or []
        try:
            log.append("审校：FAIL")
            log.append(_truncate(json.dumps(issues, ensure_ascii=False, indent=2), 2000))
        except Exception:
            pass
    else:
        log.append("审校：PASS")

    title = str(plan_obj.get("title") or "数据分析报告").strip() or "数据分析报告"

    # save files
    log_text = "\n".join(log)
    (report_dir / "report.md").write_text(final_md, encoding="utf-8")
    manifest = {
        "report_id": report_id,
        "title": title,
        "created_at": float(time.time()),
        "selected_columns": selected_columns,
        "sample_rows": sample_rows,
        "stage_models": stage_models,
        "plan": plan_obj,
        "jobs": job_results,
        "insights": insights_obj,
        "artifacts": {"base_dir": base_rel, "charts": artifacts_charts, "tables": artifacts_tables},
        "process_log": log_text,
    }
    (report_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "process_log.md").write_text(log_text, encoding="utf-8")

    elapsed_ms = int((time.time() - t0) * 1000)
    log.append(f"\n✅ 完成，用时 {elapsed_ms}ms")

    return {
        "report_id": report_id,
        "title": title,
        "content": final_md,
        "log": "\n".join(log),
        "artifacts": {
            "base_dir": base_rel,
            "report_path": f"{base_rel}/report.md",
            "manifest_path": f"{base_rel}/manifest.json",
            "process_log_path": f"{base_rel}/process_log.md",
            "charts": artifacts_charts,
            "tables": artifacts_tables,
        },
        "plan": plan_obj,
        "insights": insights_obj,
        "elapsed_ms": elapsed_ms,
    }


