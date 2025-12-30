import os
import json
import time
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
from celery import Celery

from action_engine import apply_actions
from session_store import load_session_from_disk
from analysis_engine import run_analysis
from engine_report_v2 import run_report_engine_v2  # type: ignore

broker_url = os.getenv("BROKER_URL", "redis://redis:6379/0")
result_backend = os.getenv("RESULT_BACKEND", broker_url)

celery_app = Celery("radarm", broker=broker_url, backend=result_backend)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    worker_prefetch_multiplier=1,
)


def _load_session_df(session_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    仅用于 Celery worker：从磁盘加载 session，并回放 actions 得到当前 df。
    """
    sess = load_session_from_disk(session_id)
    if not sess:
        raise ValueError(f"session {session_id} 不存在")
    raw_df = sess.get("raw_df")
    actions = (sess.get("actions") or [])[: int(sess.get("cursor", 0))]
    try:
        df = apply_actions(raw_df, actions)
    except Exception:
        df = raw_df
    return df, sess


def _safe_id(s: str) -> str:
    return str(s or "default").replace("/", "_").replace("\\", "_")[:128]


@celery_app.task(name="radarm.debug_ping")
def debug_ping(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """轻量心跳任务，用于验证 Celery 与 Redis 是否正常。"""
    return {"ok": True, "payload": payload or {}}


@celery_app.task(name="radarm.analysis")
def run_analysis_task(
    session_id: str,
    analysis: str,
    params: Dict[str, Any] | None = None,
    explain: bool = False,
    api_keys: Dict[str, Any] | None = None,
    provider: str = "deepseekA",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Celery 异步分析任务：返回图表/表格路径和摘要。
    """
    t0 = time.time()
    df, sess = _load_session_df(session_id)
    res = run_analysis(session_id=session_id, df=df, analysis=analysis, params=params or {})

    tables_obj = res.get("tables") or []
    charts_obj = res.get("charts") or []
    tables: List[Dict[str, Any]] = []
    for t in tables_obj:
        name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else "表格")
        md = getattr(t, "markdown", None) or (t.get("markdown") if isinstance(t, dict) else "")
        tables.append({"name": name, "markdown": md})
    charts: List[Dict[str, Any]] = []
    for c in charts_obj:
        name = getattr(c, "name", None) or (c.get("name") if isinstance(c, dict) else "图")
        path = getattr(c, "path", None) or (c.get("path") if isinstance(c, dict) else None)
        if path:
            charts.append({"name": name, "path": path})

    elapsed_ms = int((time.time() - t0) * 1000)
    return {
        "session_id": session_id,
        "analysis": analysis,
        "title": res.get("title", analysis),
        "tables": tables,
        "charts": charts,
        "summary": res.get("summary") or {},
        "elapsed_ms": elapsed_ms,
    }


@celery_app.task(name="radarm.report")
def run_report_task(session_id: str, req_obj: Dict[str, Any], report_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Celery 报告生成任务，复用 report v2 工作流。
    req_obj 来源于前端请求字典。
    """
    df, sess = _load_session_df(session_id)

    # 列选择 + @ 引用（精简版，与 backend 对齐）
    selected_cols: List[str] = []
    try:
        for c in req_obj.get("selectedColumns") or []:
            if c is not None:
                selected_cols.append(str(c))
    except Exception:
        pass

    cols_exist = [str(c) for c in getattr(df, "columns", [])]
    filtered_cols: List[str] = []
    seen = set()
    for c in selected_cols:
        if c in cols_exist and c not in seen:
            filtered_cols.append(c)
            seen.add(c)

    df_use = df
    if filtered_cols:
        try:
            df_use = df[filtered_cols].copy()
        except Exception:
            df_use = df

    sample_rows = None
    if req_obj.get("sampleRows") is not None:
        try:
            n = int(req_obj.get("sampleRows"))
            if n > 0 and len(df_use) > n:
                sample_rows = n
                df_use = df_use.sample(n=n, random_state=42)
        except Exception:
            sample_rows = None

    def _try_desc(d: pd.DataFrame) -> str:
        try:
            return d.describe().to_markdown()
        except Exception:
            try:
                return str(d.describe())
            except Exception:
                return ""

    data_context = (
        f"来源: {sess.get('filename','data.csv')}\n"
        f"规模: {len(df_use)} 行, {len(df_use.columns)} 列\n"
        f"字段(预览): {', '.join([str(c) for c in df_use.columns[:80]])}\n"
        f"统计摘要(数值列):\n{_try_desc(df_use)}"
    )

    report_id = _safe_id(report_id or req_obj.get("reportId") or req_obj.get("report_id") or _safe_id(time.time()))
    result = run_report_engine_v2(
        session_id=session_id,
        report_id=report_id,
        user_request=req_obj.get("userRequest", ""),
        data_context=data_context,
        api_keys=req_obj.get("apiKeys") or {},
        model_config=req_obj.get("model_config") or {},
        df=df_use,
        stage_models=req_obj.get("reportStages") or {},
        selected_columns=filtered_cols,
        sample_rows=sample_rows,
        check_cancelled=None,
    )

    artifacts = result.get("artifacts") or {}
    charts = artifacts.get("charts") or []
    tables = artifacts.get("tables") or []

    return {
        "session_id": session_id,
        "report_id": report_id,
        "title": result.get("title"),
        "content": result.get("content"),
        "charts": charts,
        "tables": tables,
        "process_log": result.get("log"),
        "manifest_path": artifacts.get("manifest_path"),
        "report_path": artifacts.get("report_path"),
    }

