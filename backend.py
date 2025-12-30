# 必须在导入 NumPy/SciPy 之前设置，禁用 Fortran 运行时的 CTRL+C 处理
# 这可以避免按 CTRL+C 退出时出现 "forrtl: error (200)" 错误
import os
os.environ.setdefault('FOR_DISABLE_CONSOLE_CTRL_HANDLER', '1')

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import base64
import matplotlib
import json
import numpy as np
import urllib.parse
from sqlalchemy import create_engine
import re
import time
import shutil
import zipfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from scipy import stats
import importlib

from action_engine import (
    action_to_brief,
    actions_to_pandas_code,
    apply_actions,
    build_data_profile,
    profile_to_context,
    validate_actions,
)
from engine_clean_actions import suggest_clean_actions
from session_store import ensure_storage_ready, load_session_from_disk, save_session_to_disk
from analysis_engine import run_analysis
from engine_report_v2 import run_report_engine_v2
from tasks import celery_app
try:
    from backend.app.services.sandbox_service import SandboxService
except Exception:
    SandboxService = None

# 引入基础工具和三个工作流
import ml_engine as ml
import workflow_single_chat
import workflow_multi_chat
import workflow_report # 这就是刚才更新的5阶段引擎
import vision_engine

# 后端绘图配置
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

app = FastAPI(title="Radarm API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 状态管理 ---
sessions = {}
# 跟踪正在生成的报告：{session_id: {report_id: True/False}}
report_generation_tracking: Dict[str, Dict[str, bool]] = {}
# 沙箱实例
sandbox_service = SandboxService() if SandboxService else None

ensure_storage_ready()


# --- 健康检查端点 ---
@app.get("/health")
async def health_check():
    """健康检查端点，用于Docker和负载均衡器"""
    return {"status": "ok", "service": "radarm"}


def _placeholder_df() -> pd.DataFrame:
    return pd.DataFrame({"info": ["请先上传数据"]})


def _history_info(session: Dict[str, Any]) -> Dict[str, Any]:
    actions = session.get("actions") or []
    cursor = int(session.get("cursor", 0))
    total = int(len(actions))
    cursor = max(0, min(cursor, total))
    return {
        "cursor": cursor,
        "total": total,
        "can_undo": cursor > 0,
        "can_redo": cursor < total,
    }


def _ensure_meta(session: Dict[str, Any]) -> Dict[str, Any]:
    meta = session.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    cols_meta = meta.get("columns")
    if not isinstance(cols_meta, dict):
        cols_meta = {}
    meta["columns"] = cols_meta
    session["meta"] = meta
    return meta


def _compute_column_maps(session: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    计算“原始列名 -> 当前列名”的映射，以及反向映射（当前 -> 原始）。
    - meta 存储按“原始列名”保存，显示/编辑按“当前列名”映射
    - 目前仅处理 rename_columns / drop_columns（已覆盖 M2 需求）
    """
    raw_df = session.get("raw_df")
    raw_cols = [str(c) for c in getattr(raw_df, "columns", [])] if raw_df is not None else []

    cursor = int(session.get("cursor", 0))
    actions = (session.get("actions") or [])[: max(0, cursor)]

    orig_to_cur: Dict[str, str] = {c: c for c in raw_cols}
    for a in actions:
        t = (a or {}).get("type")
        p = (a or {}).get("params") or {}

        if t == "rename_columns":
            mapping = p.get("mapping") or {}
            if isinstance(mapping, dict):
                for old, new in mapping.items():
                    if not old or not new:
                        continue
                    for orig, cur in list(orig_to_cur.items()):
                        if cur == old:
                            orig_to_cur[orig] = str(new)

        if t == "drop_columns":
            cols = p.get("columns") or []
            if isinstance(cols, str):
                cols = [cols]
            drop_set = set([str(x) for x in cols if x is not None])
            if drop_set:
                for orig in list(orig_to_cur.keys()):
                    if orig_to_cur.get(orig) in drop_set:
                        del orig_to_cur[orig]

    cur_to_orig: Dict[str, str] = {}
    for orig, cur in orig_to_cur.items():
        if cur not in cur_to_orig:
            cur_to_orig[cur] = orig

    return {"orig_to_cur": orig_to_cur, "cur_to_orig": cur_to_orig}


def _meta_current(session: Dict[str, Any]) -> Dict[str, Any]:
    meta = _ensure_meta(session)
    cols_meta: Dict[str, Any] = meta.get("columns", {})
    maps = _compute_column_maps(session)
    cur_to_orig = maps.get("cur_to_orig", {})

    df = session.get("df")
    current_cols = [str(c) for c in getattr(df, "columns", [])] if df is not None else []

    out: Dict[str, Any] = {}
    for cur in current_cols:
        orig = cur_to_orig.get(cur, cur)
        m = cols_meta.get(orig)
        if not isinstance(m, dict):
            m = {"label": "", "measure": "scale", "value_labels": {}, "missing_codes": []}
        # 确保字段齐全
        out[cur] = {
            "label": m.get("label", ""),
            "measure": m.get("measure", "scale"),
            "value_labels": m.get("value_labels", {}) if isinstance(m.get("value_labels", {}), dict) else {},
            "missing_codes": m.get("missing_codes", []) if isinstance(m.get("missing_codes", []), list) else [],
            "_orig": orig,
        }
    return {"columns": out, "maps": maps}


def _history_stack(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    actions = session.get("actions") or []
    cursor = int(session.get("cursor", 0))
    stack: List[Dict[str, Any]] = []
    for idx, a in enumerate(actions):
        stack.append(
            {
                "index": idx + 1,
                "applied": idx < cursor,
                "brief": action_to_brief(a),
                "action": a,
            }
        )
    return stack


def _ensure_reports(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    reports = session.get("reports")
    if not isinstance(reports, list):
        reports = []
    session["reports"] = reports
    return reports


def _safe_id(s: str) -> str:
    t = re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(s or "").strip())
    t = t.replace("..", "_")[:128]
    return t or "default"


def _report_dir(session_id: str, report_id: str) -> Path:
    sid = _safe_id(session_id)
    rid = _safe_id(report_id)
    return Path("out") / sid / "reports" / rid


def _report_meta_from_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    # 只返回轻量字段给前端列表
    return {
        "report_id": entry.get("report_id"),
        "title": entry.get("title"),
        "created_at": entry.get("created_at"),
        "user_request": entry.get("user_request", ""),
        "selected_columns": entry.get("selected_columns") or [],
        "sample_rows": entry.get("sample_rows"),
        "base_dir": entry.get("base_dir"),
        "charts": entry.get("charts") or [],
    }


def _recompute_current_df(session: Dict[str, Any]) -> None:
    raw_df = session.get("raw_df")
    if raw_df is None:
        raw_df = _placeholder_df()
        session["raw_df"] = raw_df

    actions = session.get("actions") or []
    cursor = int(session.get("cursor", 0))
    cursor = max(0, min(cursor, len(actions)))
    session["cursor"] = cursor

    try:
        current_df = apply_actions(raw_df, actions[:cursor])
    except Exception:
        # 若 action 回放失败，回退到 raw_df，避免 session 彻底不可用
        current_df = raw_df.copy()
        session["actions"] = []
        session["cursor"] = 0

    session["df"] = current_df
    prof = build_data_profile(current_df)
    session["profile"] = prof
    session["profile_context"] = profile_to_context(prof)


def get_session_data(session_id: str) -> Dict[str, Any]:
    if session_id in sessions:
        return sessions[session_id]

    loaded = load_session_from_disk(session_id)
    if loaded:
        session = {
            "version": loaded.get("version", 1),
            "session_id": session_id,
            "filename": loaded.get("filename", "data.csv"),
            "created_at": loaded.get("created_at", time.time()),
            "updated_at": loaded.get("updated_at", time.time()),
            "meta": loaded.get("meta", {}) or {"columns": {}},
            "reports": loaded.get("reports", []) or [],
            "agent_pending": loaded.get("agent_pending", None),
            "actions": loaded.get("actions", []) or [],
            "cursor": int(loaded.get("cursor", 0)),
            "raw_df": loaded.get("raw_df") if loaded.get("raw_df") is not None else _placeholder_df(),
        }
        _ensure_meta(session)
        _recompute_current_df(session)
        sessions[session_id] = session
        return session

    # 新 session
    now = time.time()
    session = {
        "version": 1,
        "session_id": session_id,
        "filename": "data.csv",
        "created_at": now,
        "updated_at": now,
        "meta": {"columns": {}},
        "reports": [],
        "agent_pending": None,
        "actions": [],
        "cursor": 0,
        "raw_df": _placeholder_df(),
    }
    _ensure_meta(session)
    _recompute_current_df(session)
    sessions[session_id] = session
    save_session_to_disk(session)
    return session


def update_session_data(session_id: str, df: pd.DataFrame, filename: str = None):
    """
    将新的 df 作为“新的 raw 基线”写入 session（会清空 action 历史）。
    说明：旧的聊天/代码执行模式可能直接修改 df，这里用“提交为新基线”保证可持久化恢复。
    """
    session = get_session_data(session_id)
    session["raw_df"] = df.copy()
    session["df"] = df.copy()
    session["actions"] = []
    session["cursor"] = 0
    # 新数据集：重置列元数据
    session["meta"] = {"columns": {}}
    # 新数据集：清空历史报告（避免旧报告与新数据不一致）
    session["reports"] = []
    session["agent_pending"] = None
    if filename:
        session["filename"] = filename
    _ensure_meta(session)
    _recompute_current_df(session)
    save_session_to_disk(session)
    sessions[session_id] = session

# --- 数据模型 ---
class ChatRequest(BaseModel):
    session_id: str
    message: str
    # 前端会传入多个 provider 的 key（DeepSeekA/B/C + zhipu + qwen）
    apiKeys: Dict[str, str] = {}
    # ask | agent_single | agent_multi （兼容旧值：single/expert_mixed）
    mode: str = "agent_single"
    # ask / agent_single：{ provider: "...", model: "..." }
    modelSelection: Dict[str, Any] = {}
    # agent_multi：{ planner:{provider,model}, executor:{...}, verifier:{...} }
    agentRoles: Dict[str, Any] = {}
    # 联网搜索开关 + 图片附件（路径）（Ask / Agent 均可用）
    webSearch: bool = False
    imagePaths: List[str] = []
    # 视觉理解（GLM-4V / Qwen-VL）：把图片转成“文字描述”注入上下文，提升看图能力
    # - visionEnabled: 是否启用（默认启用；若没有 zhipu/qwen key 会自动跳过）
    # - visionProvider: auto | zhipu | qwen
    # - visionModel: 可选覆盖模型名（会覆盖当前选择的 visionProvider 对应默认模型）
    visionEnabled: bool = True
    visionProvider: str = "auto"
    visionModel: Optional[str] = None
    dataContext: str = ""

class ReportRequest(BaseModel):
    session_id: str
    apiKeys: Dict[str, str] = {}
    # 兼容旧字段（忽略）：mode/primaryModel
    mode: str = "report"
    primaryModel: Optional[str] = None
    userRequest: str = ""  # 用户的具体分析需求（可空）
    # 选取部分数据（列/抽样）
    selectedColumns: List[str] = []
    sampleRows: Optional[int] = None
    # 多模型工作流配置（可选）：{ planner:{provider,model}, analyst:{...}, writer:{...}, reviewer:{...} }
    reportStages: Dict[str, Any] = {}
    # 多份报告：saveAsNew=true 默认生成新 report_id；否则可覆盖 reportId
    reportId: Optional[str] = None
    saveAsNew: bool = True
    runAsync: bool = False

class DBConnectRequest(BaseModel):
    session_id: str
    type: str
    host: str = ""
    port: str = ""
    user: str = ""
    password: str = ""
    database: str = ""
    sql: str = "SELECT * FROM users LIMIT 100"

class ResetRequest(BaseModel):
    session_id: str

class ReportCancelRequest(BaseModel):
    session_id: str
    report_id: Optional[str] = None

class ReportDownloadRequest(BaseModel):
    content: str
    filename: str


@app.get("/tasks/{task_id}")
async def task_status(task_id: str):
    """
    查询 Celery 任务状态/结果。
    """
    res = celery_app.AsyncResult(task_id)
    result = None
    if res.successful():
        try:
            result = res.get()
        except Exception:
            result = str(res.result)
    elif res.failed():
        result = str(res.result)
    return {
        "id": task_id,
        "state": res.state,
        "ready": res.ready(),
        "successful": res.successful(),
        "result": result,
    }

class ApplyActionsRequest(BaseModel):
    session_id: str
    actions: List[Dict[str, Any]]
    # 可选：用于“应用后继续 Agent 任务”
    apiKeys: Dict[str, str] = {}
    mode: str = "agent_single"  # agent_single | agent_multi
    modelSelection: Dict[str, Any] = {}
    agentRoles: Dict[str, Any] = {}
    autoContinue: bool = True


class SetCursorRequest(BaseModel):
    session_id: str
    cursor: int


class ColumnMetaRequest(BaseModel):
    session_id: str
    column: str
    label: str = ""
    measure: str = "scale"  # scale|nominal|ordinal
    value_labels: Dict[str, str] = {}
    missing_codes: List[Any] = []


class AnalysisRequest(BaseModel):
    session_id: str
    analysis: str
    params: Dict[str, Any] = {}
    apiKeys: Dict[str, str] = {}
    # 优先使用 provider/model；兼容旧字段 primaryModel
    provider: str = "deepseekA"
    model: Optional[str] = None
    primaryModel: Optional[str] = None
    explain: bool = True
    runAsync: bool = False


class OnboardSuggestRequest(BaseModel):
    session_id: str
    apiKeys: Dict[str, str] = {}
    # 期望使用的 provider/model（如果该 provider 没有 key，后端会自动兜底到其他已配置 provider）
    provider: str = "deepseekA"
    model: Optional[str] = None

# --- 配置 ---
# 支持多 provider：
# - DeepSeek：三套 API 槽位（A/B/C）
# - Zhipu：单槽位
# - Qwen：单槽位（DashScope OpenAI-compatible）
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions")
ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")

# --- 视觉理解（GLM-4V / Qwen-VL / Qwen-Omni）---
# 说明：视觉模型用于“看图生成文字描述”，再注入到 Ask/Agent 的文本上下文中
ZHIPU_VISION_MODEL = os.getenv("ZHIPU_VISION_MODEL", "glm-4v")
# Qwen 视觉/全模态模型示例：qwen-vl-plus / qwen-omni-turbo / qwen3-omni-flash
QWEN_VISION_MODEL = os.getenv("QWEN_VISION_MODEL", "qwen-vl-plus")
VISION_MAX_IMAGES = int(os.getenv("VISION_MAX_IMAGES", "3"))
VISION_MAX_EDGE = int(os.getenv("VISION_MAX_EDGE", "1024"))
VISION_MAX_BYTES = int(os.getenv("VISION_MAX_BYTES", "2000000"))
VISION_TIMEOUT = int(os.getenv("VISION_TIMEOUT", "120"))

MODEL_CONFIG = {
    "deepseekA": {"url": os.getenv("DEEPSEEK_A_URL", DEEPSEEK_BASE_URL), "model": os.getenv("DEEPSEEK_A_MODEL", "deepseek-reasoner")},
    "deepseekB": {"url": os.getenv("DEEPSEEK_B_URL", DEEPSEEK_BASE_URL), "model": os.getenv("DEEPSEEK_B_MODEL", "deepseek-reasoner")},
    "deepseekC": {"url": os.getenv("DEEPSEEK_C_URL", DEEPSEEK_BASE_URL), "model": os.getenv("DEEPSEEK_C_MODEL", "deepseek-reasoner")},
    "zhipu": {"url": os.getenv("ZHIPU_URL", ZHIPU_BASE_URL), "model": os.getenv("ZHIPU_MODEL", "glm-4.7")},
    "qwen": {"url": os.getenv("QWEN_URL", QWEN_BASE_URL), "model": os.getenv("QWEN_MODEL", "qwen-max")},
}
# 自动清洗 URL
for k in MODEL_CONFIG:
    m = re.search(r'\((https?://.*?)\)', MODEL_CONFIG[k]['url'])
    if m: MODEL_CONFIG[k]['url'] = m.group(1)
    else: MODEL_CONFIG[k]['url'] = re.sub(r'[\[\]\(\)\s\']', '', MODEL_CONFIG[k]['url'])


def _choose_provider(api_keys: Dict[str, str], primary_model: str) -> Optional[str]:
    """
    在多个 key 中选择一个可用 provider：
    - 优先使用 primary_model（如果有 key）
    - 否则返回第一个有 key 的 provider
    """
    if api_keys.get(primary_model):
        return primary_model
    for k, v in (api_keys or {}).items():
        if v:
            return k
    return None


def _call_llm(
    provider: str,
    api_key: str,
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    timeout: int = 120,
) -> Optional[str]:
    if not api_key:
        return None
    cfg = MODEL_CONFIG.get(provider)
    if not cfg:
        return None
    try:
        requests = importlib.import_module("requests")
    except Exception:
        return None
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {"model": (model or cfg["model"]), "messages": [{"role": "user", "content": prompt}], "temperature": temperature}
    try:
        resp = requests.post(cfg["url"], headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def _normalize_mode(mode: str) -> str:
    m = (mode or "").strip().lower()
    if m in ("ask", "chat", "ask_mode"):
        return "ask"
    if m in ("agent_single", "agent", "single", "agent_single_mode"):
        return "agent_single"
    if m in ("agent_multi", "expert_mixed", "agent_multi_mode"):
        return "agent_multi"
    return "agent_single"


def _clone_model_config() -> Dict[str, Dict[str, Any]]:
    return {k: {**v} for k, v in (MODEL_CONFIG or {}).items()}


def _apply_model_overrides(model_config: Dict[str, Dict[str, Any]], overrides: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    mc = _clone_model_config() if model_config is MODEL_CONFIG else {k: {**v} for k, v in (model_config or {}).items()}
    for prov, model in (overrides or {}).items():
        if prov in mc and model:
            mc[prov]["model"] = str(model)
    return mc


def _extract_at_mentions(text: str) -> List[str]:
    """
    提取 @xxx token（直到遇到空白/常见标点）。
    """
    if not text:
        return []
    tokens = re.findall(r"@([^\s，,。;；:：\)\]\}]+)", str(text))
    seen = set()
    out = []
    for t in tokens:
        t = str(t).strip()
        if not t or t in seen:
            continue
        out.append(t)
        seen.add(t)
    return out


def _norm_name(s: str) -> str:
    return re.sub(r"[\s\-_]+", "", str(s or "").strip().lower())


def _resolve_mention_to_column(token: str, columns: List[str]) -> Optional[str]:
    cols = [str(c) for c in (columns or [])]
    if token in cols:
        return token
    nmap = {_norm_name(c): c for c in cols}
    tn = _norm_name(token)
    if tn in nmap:
        return nmap[tn]
    # substring match
    for c in cols:
        if tn and tn in _norm_name(c):
            return c
    return None


def _build_at_context(df: pd.DataFrame, tokens: List[str], *, max_rows: int = 8) -> str:
    if df is None or not tokens:
        return ""
    cols = [str(c) for c in getattr(df, "columns", [])]
    blocks = []
    for t in tokens:
        col = _resolve_mention_to_column(t, cols)
        if not col:
            blocks.append(f"- @{t}: 未找到对应列名")
            continue
        s = df[col]
        try:
            miss_pct = float(s.isna().mean() * 100) if len(df) else 0.0
        except Exception:
            miss_pct = None
        try:
            nunique = int(s.nunique(dropna=True))
        except Exception:
            nunique = None

        head = df[[col]].head(max_rows).copy()
        try:
            head_md = head.to_markdown(index=False)
        except Exception:
            head_md = head.to_string(index=False)

        extra = ""
        try:
            sn = pd.to_numeric(s, errors="coerce")
            if sn.notna().sum() >= 3:
                extra = f"数值摘要：mean={float(sn.mean()):.4f}, std={float(sn.std()):.4f}, min={float(sn.min()):.4f}, max={float(sn.max()):.4f}"
        except Exception:
            extra = ""

        blocks.append(
            f"- @{t} → 列名 `{col}` (dtype={getattr(s, 'dtype', '')}, missing%={None if miss_pct is None else round(miss_pct,2)}, nunique={nunique})\n"
            + (f"  - {extra}\n" if extra else "")
            + "  - 预览：\n"
            + head_md
        )
    return "\n\n".join(blocks)


def _maybe_parse_quick_analysis_request(text: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    尝试把用户在聊天中提出的“SPSS 风格分析需求”解析为 analysis_engine 的调用参数。

    返回：
      None 或 { "analysis": str, "params": dict, "title": str }

    设计目标：
    - 不依赖 LLM（更快、更稳定）
    - 只在高度确定时触发；否则返回 None 交给原 Agent 工作流处理
    """
    msg = str(text or "").strip()
    if not msg:
        return None

    low = msg.lower()
    cols_all = [str(c) for c in getattr(df, "columns", [])]

    # 1) 优先用 @ 引用列
    cols: List[str] = []
    try:
        for t in _extract_at_mentions(msg):
            c = _resolve_mention_to_column(t, cols_all)
            if c and c not in cols:
                cols.append(c)
    except Exception:
        cols = []

    # 2) 若未 @，尝试在文本里直接匹配列名（轻量 substring）
    if not cols:
        tn = _norm_name(msg)
        for c in cols_all:
            cn = _norm_name(c)
            if cn and cn in tn:
                cols.append(c)
                if len(cols) >= 6:
                    break

    # --- 类型识别：尽量保守 ---
    if ("数据概览" in msg) or ("overview" in low) or (("概览" in msg) and ("数据" in msg or "表" in msg)):
        return {"analysis": "overview", "params": {}, "title": "数据概览"}

    if ("频数" in msg) or ("频率" in msg) or ("频次" in msg):
        if not cols:
            return None
        return {"analysis": "frequency", "params": {"column": cols[0], "top_n": 20}, "title": f"频数分析：{cols[0]}"}

    if ("列联" in msg) or ("交叉表" in msg) or ("crosstab" in low) or (("交叉" in msg) and len(cols) >= 2):
        if len(cols) < 2:
            return None
        return {
            "analysis": "crosstab",
            "params": {"row": cols[0], "col": cols[1], "normalize": False},
            "title": f"列联(交叉)分析：{cols[0]} × {cols[1]}",
        }

    if ("正态" in msg) or ("正态性" in msg) or ("normality" in low):
        if not cols:
            return None
        return {"analysis": "normality", "params": {"column": cols[0]}, "title": f"正态性检验：{cols[0]}"}

    if ("描述统计" in msg) or ("描述性统计" in msg) or ("descriptive" in low):
        use_cols = cols
        if not use_cols:
            try:
                use_cols = [str(c) for c in df.select_dtypes(include=[np.number]).columns][:8]
            except Exception:
                use_cols = []
        if not use_cols:
            return None
        return {"analysis": "descriptive", "params": {"columns": use_cols}, "title": "描述性统计"}

    if ("分类汇总" in msg) or (("分组" in msg or "按" in msg) and ("汇总" in msg or "统计" in msg)):
        if len(cols) < 2:
            return None
        # 约定：第 1 个是分组列，第 2 个是数值列
        return {
            "analysis": "group_summary",
            "params": {"group_by": cols[0], "metric": cols[1], "agg": "mean"},
            "title": f"分类汇总：{cols[0]} → mean({cols[1]})",
        }

    if ("t检验" in msg) or ("t-test" in low) or ("ttest" in low):
        # 约定：
        # - 单样本：@y（可选 mu=）
        # - 配对：@y @y2
        # - 独立样本：@y @group_col（可选 @group_a @group_b）
        if not cols:
            return None
        if ("单样本" in msg) or ("one" in low and "sample" in low):
            mu = 0.0
            mmu = re.search(r"(?:mu|均值|检验值)\s*=?\s*(-?\d+(?:\.\d+)?)", low)
            if mmu:
                try:
                    mu = float(mmu.group(1))
                except Exception:
                    mu = 0.0
            return {"analysis": "ttest", "params": {"ttype": "one_sample", "y": cols[0], "mu": mu}, "title": f"单样本T检验：{cols[0]}（mu={mu}）"}
        if ("配对" in msg) or ("paired" in low):
            if len(cols) < 2:
                return None
            return {"analysis": "ttest", "params": {"ttype": "paired", "y": cols[0], "y2": cols[1]}, "title": f"配对样本T检验：{cols[0]} vs {cols[1]}"}
        # 默认独立样本
        if len(cols) < 2:
            return None
        return {"analysis": "ttest", "params": {"ttype": "independent", "y": cols[0], "group_col": cols[1]}, "title": f"独立样本T检验：{cols[0]} by {cols[1]}"}

    if ("方差分析" in msg) or ("anova" in low):
        if len(cols) < 2:
            return None
        return {"analysis": "anova", "params": {"y": cols[0], "group_col": cols[1]}, "title": f"单因素方差分析：{cols[0]} by {cols[1]}"}

    if ("卡方" in msg) or ("chi-square" in low) or ("chi2" in low):
        if len(cols) < 2:
            return None
        return {"analysis": "chi_square", "params": {"row": cols[0], "col": cols[1]}, "title": f"卡方检验：{cols[0]} × {cols[1]}"}

    if ("非参" in msg) or ("mann" in low) or ("kruskal" in low) or ("friedman" in low):
        # 支持 3 个常用：Mann-Whitney / Kruskal-Wallis / Friedman
        test = ""
        if "mann" in low or "mw" in low or "mann-whitney" in low or "u检验" in msg:
            test = "mannwhitney"
        elif "kruskal" in low or "k-w" in low or "kw" in low:
            test = "kruskal"
        elif "friedman" in low:
            test = "friedman"
        if not test:
            return None
        if test == "friedman":
            if len(cols) < 2:
                return None
            return {"analysis": "nonparam", "params": {"test": test, "columns": cols[:6]}, "title": f"Friedman 非参数检验（k={min(len(cols),6)}）"}
        # mann/kruskal：需要 y + group_col
        if len(cols) < 2:
            return None
        return {"analysis": "nonparam", "params": {"test": test, "y": cols[0], "group_col": cols[1]}, "title": f"{test} 非参数检验：{cols[0]} by {cols[1]}"}

    if ("相关" in msg) or ("correlation" in low):
        use_cols = cols
        if len(use_cols) < 2:
            try:
                use_cols = [str(c) for c in df.select_dtypes(include=[np.number]).columns][:6]
            except Exception:
                use_cols = []
        if len(use_cols) < 2:
            return None
        method = "pearson"
        if ("spearman" in low) or ("斯皮尔曼" in msg):
            method = "spearman"
        if ("kendall" in low) or ("肯德尔" in msg):
            method = "kendall"
        return {"analysis": "correlation", "params": {"columns": use_cols, "method": method}, "title": f"{method.title()} 相关性分析"}

    if ("逻辑回归" in msg) or ("logistic" in low):
        if len(cols) < 2:
            return None
        return {"analysis": "logistic_regression", "params": {"y": cols[0], "x": cols[1:]}, "title": f"逻辑回归：{cols[0]} ~ {', '.join(cols[1:])}"}

    if ("线性回归" in msg) or ("ols" in low) or (("回归" in msg) and ("logistic" not in low) and ("逻辑" not in msg)):
        if len(cols) < 2:
            return None
        return {"analysis": "linear_regression", "params": {"y": cols[0], "x": cols[1:]}, "title": f"线性回归：{cols[0]} ~ {', '.join(cols[1:])}"}

    if ("pca" in low) or ("主成分" in msg):
        use_cols = cols
        if len(use_cols) < 2:
            try:
                use_cols = [str(c) for c in df.select_dtypes(include=[np.number]).columns][:6]
            except Exception:
                use_cols = []
        if len(use_cols) < 2:
            return None
        m = re.search(r"(?:组件|成分|维度)\s*(\d+)", msg)
        n_components = int(m.group(1)) if m else 2
        return {"analysis": "pca", "params": {"columns": use_cols, "n_components": n_components}, "title": f"PCA（n_components={n_components}）"}

    if ("kmeans" in low) or ("k-means" in low) or ("聚类" in msg):
        use_cols = cols
        if len(use_cols) < 2:
            try:
                use_cols = [str(c) for c in df.select_dtypes(include=[np.number]).columns][:6]
            except Exception:
                use_cols = []
        if len(use_cols) < 2:
            return None
        mk = re.search(r"(?:k\s*=\s*|k\s*)(\d+)", low)
        k = int(mk.group(1)) if mk else 3
        return {"analysis": "kmeans", "params": {"columns": use_cols, "k": k}, "title": f"KMeans（k={k}）"}

    return None


def _dataset_health_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    导入后“预检”用的轻量摘要（不依赖 LLM）。
    仅用于给 LLM/用户提示，不作为严谨统计结论。
    """
    rows = int(len(df))
    cols = int(len(getattr(df, "columns", [])))
    out: Dict[str, Any] = {"shape": {"rows": rows, "cols": cols}}

    # 复用 pandas dtype 分类（更像 SPSS 的变量视图）
    type_count = {"numeric": 0, "categorical": 0, "datetime": 0, "bool": 0, "other": 0}
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    binary_cols: List[str] = []
    id_like_cols: List[str] = []
    constant_cols: List[str] = []
    datetime_like_cols: List[str] = []

    dup_pct = None
    try:
        dup_pct = float(df.duplicated().mean() * 100) if rows > 0 else 0.0
    except Exception:
        dup_pct = None

    out["dup_pct"] = None if dup_pct is None else round(dup_pct, 2)

    cols_all = [str(c) for c in getattr(df, "columns", [])]
    for c in cols_all:
        s = df[c]
        try:
            nunique = int(s.nunique(dropna=True))
        except Exception:
            nunique = None
        try:
            miss_pct = float(s.isna().mean() * 100) if rows > 0 else 0.0
        except Exception:
            miss_pct = None

        # 常量列
        if nunique == 1:
            constant_cols.append(c)

        # dtype 分类
        try:
            if pd.api.types.is_datetime64_any_dtype(s):
                type_count["datetime"] += 1
                datetime_like_cols.append(c)
            elif pd.api.types.is_bool_dtype(s):
                type_count["bool"] += 1
                binary_cols.append(c) if (nunique == 2) else None
            elif pd.api.types.is_numeric_dtype(s):
                type_count["numeric"] += 1
                numeric_cols.append(c)
                # 低基数数值列也可能是分类/分组
                if nunique is not None and nunique <= 12 and (miss_pct is None or miss_pct <= 80):
                    categorical_cols.append(c)
            else:
                # object/category/string
                type_count["categorical"] += 1
                categorical_cols.append(c)
        except Exception:
            type_count["other"] += 1

        # 二分类列（更可能作为 y 或分组）
        if nunique == 2:
            if c not in binary_cols:
                binary_cols.append(c)

        # ID-like：唯一值接近行数
        try:
            if rows > 0 and nunique is not None:
                if nunique >= max(20, int(rows * 0.9)):
                    id_like_cols.append(c)
        except Exception:
            pass

        # datetime-like（基于列名）
        name_low = str(c).lower()
        if any(k in name_low for k in ["date", "time", "datetime", "timestamp"]) or any(k in str(c) for k in ["日期", "时间", "年月", "日", "月", "年"]):
            if c not in datetime_like_cols:
                datetime_like_cols.append(c)

    # 去重保序 + 截断，避免 prompt 过大
    def _dedup(xs: List[str], n: int) -> List[str]:
        seen = set()
        out2 = []
        for x in xs:
            if x in seen:
                continue
            out2.append(x)
            seen.add(x)
            if len(out2) >= n:
                break
        return out2

    out["type_count"] = type_count
    out["numeric_cols"] = _dedup(numeric_cols, 12)
    out["categorical_cols"] = _dedup(categorical_cols, 12)
    out["binary_cols"] = _dedup(binary_cols, 12)
    out["id_like_cols"] = _dedup(id_like_cols, 8)
    out["constant_cols"] = _dedup(constant_cols, 8)
    out["datetime_like_cols"] = _dedup(datetime_like_cols, 8)
    return out


def _duckduckgo_search(query: str, *, max_results: int = 5) -> List[Dict[str, str]]:
    """
    轻量联网搜索（无需 key）：抓取 DuckDuckGo HTML（可能受限，失败则返回空列表）。
    """
    q = (query or "").strip()
    if not q:
        return []
    try:
        requests = importlib.import_module("requests")
    except Exception:
        return []
    try:
        resp = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": q},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        html = resp.text or ""
        # result__a + result__snippet（正则轻解析）
        links = re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, flags=re.IGNORECASE)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, flags=re.IGNORECASE)

        def _strip_tags(s: str) -> str:
            s = re.sub(r"<[^>]+>", "", s or "")
            s = re.sub(r"\s+", " ", s).strip()
            return s

        results = []
        for i, (url, title_html) in enumerate(links[: max_results * 2]):
            title = _strip_tags(title_html)
            snippet = _strip_tags(snippets[i]) if i < len(snippets) else ""
            if not title:
                continue
            results.append({"title": title, "url": url, "snippet": snippet})
            if len(results) >= max_results:
                break
        return results
    except Exception:
        return []


def _try_ocr_images(image_paths: List[str]) -> Dict[str, Any]:
    """
    尝试对图片做 OCR（可选）：如果缺少依赖则返回空/错误信息。
    """
    out: Dict[str, Any] = {"images": image_paths or [], "ocr": []}
    if not image_paths:
        return out
    try:
        from PIL import Image  # type: ignore
    except Exception:
        out["note"] = "缺少 Pillow，无法读取图片做 OCR。"
        return out
    try:
        import pytesseract  # type: ignore
    except Exception:
        out["note"] = "缺少 pytesseract 或系统未安装 tesseract，无法 OCR（你仍可配置视觉模型后再支持）。"
        return out

    for p in image_paths[:3]:
        try:
            fp = Path("out") / p if not str(p).startswith("out") else Path(p)
            if not fp.exists():
                continue
            img = Image.open(fp)
            text = pytesseract.image_to_string(img)
            text = (text or "").strip()
            if text:
                out["ocr"].append({"path": p, "text": text[:2000]})
        except Exception:
            continue
    return out


def _truncate_text(s: Any, n: int = 2000) -> str:
    t = str(s or "")
    return t if len(t) <= n else (t[:n] + "\n...(截断)")


def _vision_understand_images(
    *,
    session: Dict[str, Any],
    user_query: str,
    image_paths: List[str],
    api_keys: Dict[str, str],
    vision_enabled: bool = True,
    vision_provider: str = "qwen",  # 固定使用 qwen
    vision_model_override: Optional[str] = "qwen-omni-turbo",  # 固定使用 qwen-omni-turbo
) -> Dict[str, Any]:
    """
    使用 GLM-4V / Qwen-VL 对图片生成“文字描述”，并做 session 级缓存（避免每条消息重复看图）。

    返回：
    {
      enabled: bool,
      provider: "zhipu"|"qwen"|None,
      model: str|None,
      results: [{path, ok, text?, error?, cached?}],
      truncated: bool,
      note: str
    }
    """
    enabled = bool(vision_enabled)
    paths = [str(p) for p in (image_paths or []) if str(p or "").strip()]
    if not enabled or not paths:
        return {"enabled": enabled, "provider": None, "model": None, "results": [], "truncated": False, "note": "disabled_or_no_images"}

    max_n = max(0, int(VISION_MAX_IMAGES))
    use_paths = paths[:max_n] if max_n else []
    truncated = len(paths) > len(use_paths)

    # 固定使用 qwen provider 和 qwen-omni-turbo 模型
    provider_used = "qwen"
    expected_model = "qwen-omni-turbo"
    
    # 检查 qwen API key
    if not (api_keys or {}).get("qwen"):
        return {
            "enabled": enabled,
            "provider": None,
            "model": None,
            "results": [],
            "truncated": truncated,
            "note": "missing_vision_key",
        }

    # session 级缓存：{ "sid/xxx.png": {provider, model, text, ts} }
    cache: Dict[str, Any] = session.setdefault("vision_cache", {}) if isinstance(session, dict) else {}

    cached_map: Dict[str, Dict[str, Any]] = {}
    need: List[str] = []
    for p in use_paths:
        c = cache.get(p) if isinstance(cache, dict) else None
        if isinstance(c, dict) and c.get("provider") == provider_used and c.get("model") == expected_model and c.get("text"):
            cached_map[p] = {"path": p, "ok": True, "text": str(c.get("text")), "cached": True}
        else:
            need.append(p)

    computed_map: Dict[str, Dict[str, Any]] = {}
    pack: Dict[str, Any] = {}
    if need:
        try:
            pack = vision_engine.describe_images(
                image_paths=need,
                user_query=user_query or "",
                api_keys=api_keys or {},
                preference="qwen",  # 固定使用 qwen
                zhipu_url=ZHIPU_BASE_URL,
                qwen_url=QWEN_BASE_URL,
                zhipu_model="glm-4v",  # 不再使用，保留以兼容接口
                qwen_model="qwen-omni-turbo",  # 固定使用
                max_images=len(need),
                max_edge=int(VISION_MAX_EDGE),
                max_bytes=int(VISION_MAX_BYTES),
                timeout=int(VISION_TIMEOUT),
            )
        except Exception as e:
            return {
                "enabled": enabled,
                "provider": provider_used,
                "model": expected_model,
                "results": [{"path": p, "ok": False, "error": f"vision_exception: {str(e)}"} for p in use_paths],
                "truncated": truncated,
                "note": "vision_exception",
            }

        # 固定使用 qwen-omni-turbo，不再需要 fallback 逻辑

        for item in (pack.get("results") or []):
            if not isinstance(item, dict) or not item.get("path"):
                continue
            p = str(item.get("path"))
            computed_map[p] = item
            if item.get("ok") and item.get("text") and isinstance(cache, dict):
                cache[p] = {"provider": pack.get("provider"), "model": pack.get("model"), "text": str(item.get("text")), "ts": pack.get("ts")}

        # pack 可能提示 truncated（虽然我们只传 need，但保留语义）
        try:
            if bool(pack.get("truncated")):
                truncated = True
        except Exception:
            pass

    # 合并为原始顺序
    ordered: List[Dict[str, Any]] = []
    for p in use_paths:
        if p in cached_map:
            ordered.append(cached_map[p])
        elif p in computed_map:
            ordered.append(computed_map[p])
        else:
            ordered.append({"path": p, "ok": False, "error": "vision_missing_result"})

    prov_final = str(pack.get("provider") or provider_used) if isinstance(pack, dict) else provider_used
    model_final = str(pack.get("model") or expected_model) if isinstance(pack, dict) else expected_model

    return {
        "enabled": enabled,
        "provider": prov_final,
        "model": model_final,
        "results": ordered,
        "truncated": bool(truncated),
        "note": "ok",
    }

# --- 辅助函数 ---
def sanitize_df_for_json(df_slice):
    d = df_slice.copy()
    d = d.replace([float('inf'), float('-inf')], float('nan'))
    d = d.fillna("")
    return d.to_dict(orient='records')

def _run_code_in_sandbox(code_str: str, df: pd.DataFrame, session_id: str) -> Optional[tuple]:
    """
    优先在 Docker 沙箱中执行代码。
    返回 (output_text, img_path, new_df) 或 None 表示失败。
    """
    if not sandbox_service or not getattr(sandbox_service, "available", False):
        return None

    try:
        safe_sid = _safe_id(session_id)
        work_host = Path("radarm_data") / "workspaces" / safe_sid
        work_host.mkdir(parents=True, exist_ok=True)
        input_pkl = work_host / "input.pkl"
        output_pkl = work_host / "output.pkl"
        output_json = work_host / "output.json"
        runner_py = work_host / "run.py"

        df.copy().to_pickle(input_pkl)

        runner_code = f"""
import json, pickle, time, os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

with open("input.pkl", "rb") as f:
    df = pickle.load(f)

local_vars = {{
    "df": df.copy(),
    "pd": pd,
    "np": np,
    "json": json,
    "plt": plt,
    "sns": sns,
}}
code_str = {json.dumps(code_str)}
exec(code_str, {{}}, local_vars)

text_res = str(local_vars.get("result", "执行成功"))
print_out = ""

# 保存图像（如果有）
img_path = None
if plt.get_fignums():
    ts = int(time.time() * 1000)
    fname = f"chart_{{ts}}.png"
    plt.savefig(fname, format="png", bbox_inches="tight", dpi=300)
    img_path = fname

with open("output.pkl", "wb") as f:
    pickle.dump(local_vars.get("df", df), f)

with open("output.json", "w", encoding="utf-8") as f:
    f.write(json.dumps({{"output": text_res, "img": img_path}}, ensure_ascii=False))
"""
        runner_py.write_text(runner_code, encoding="utf-8")

        res = sandbox_service.exec(session_id, ["python", "run.py"], workdir="/workspace")
        if res.get("exit_code", -1) != 0:
            return None

        if not output_json.exists() or not output_pkl.exists():
            return None
        meta = json.loads(output_json.read_text(encoding="utf-8"))
        new_df = pd.read_pickle(output_pkl)

        img_rel = meta.get("img")
        img_path = None
        if img_rel:
            # 将沙箱内生成的图表移到 out/{session_id}/
            src = work_host / img_rel
            if src.exists():
                out_dir = Path("out") / safe_sid
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = int(time.time() * 1000)
                dst = out_dir / f"chart_{ts}.png"
                src.replace(dst)
                img_path = f"{safe_sid}/{dst.name}"
        return meta.get("output", ""), img_path, new_df
    except Exception:
        return None


def execute_code(code_str: str, df: pd.DataFrame, session_id: str = "default"):
    """
    代码执行沙盒：优先 Docker 沙箱，失败回退本地执行。
    """
    # --- 安全与稳定性：禁止文件/网络/系统操作（避免读取 data.csv 等外部文件） ---
    forbidden_patterns = [
        r"\bimport\s+os\b",
        r"\bimport\s+pathlib\b",
        r"\bos\.\w+",
        r"\bopen\s*\(",
        r"\bpd\s*\.\s*read_csv\s*\(",
        r"\bread_csv\s*\(",
        r"\bpd\s*\.\s*read_excel\s*\(",
        r"\bread_excel\s*\(",
        r"\bpd\s*\.\s*read_parquet\s*\(",
        r"\bread_parquet\s*\(",
        r"\bpd\s*\.\s*read_json\s*\(",
        r"\bread_json\s*\(",
        r"\bimport\s+requests\b",
        r"\brequests\.\w+",
        r"\burllib\.\w+",
        r"\bimport\s+urllib\b",
        r"\bsocket\b",
        r"\bsubprocess\b",
        r"\bimport\s+subprocess\b",
    ]
    for pat in forbidden_patterns:
        if re.search(pat, code_str, re.IGNORECASE):
            return "禁止文件/网络/系统操作：请直接使用已提供的 df 进行分析（不要读取 data.csv 等外部文件）。", None, df

    # --- 优先尝试 Docker 沙箱 ---
    sbx_res = _run_code_in_sandbox(code_str, df, session_id)
    if sbx_res is not None:
        return sbx_res
    
    # 预置常用依赖，减少 LLM 生成代码因 NameError 失败
    local_vars = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "json": json,
        "plt": plt,
        "sns": sns,
        "ml": ml,
    }

    # --- 预置工具函数：列名模糊匹配（提升多专家建模命中率） ---
    def _norm_col_name(s: str) -> str:
        return re.sub(r"[\s\-_]+", "", str(s or "").strip().lower())

    def find_col(*candidates: str):
        """
        在当前 df.columns 中按候选名查找真实列名；支持忽略空格/大小写与包含匹配。
        返回：真实列名(str) 或 None
        """
        _df = local_vars.get("df")
        cols = list(getattr(_df, "columns", [])) if _df is not None else []
        norm_map = {_norm_col_name(c): c for c in cols}

        # exact normalized
        for cand in candidates or []:
            cn = _norm_col_name(cand)
            if cn in norm_map:
                return norm_map[cn]

        # substring
        for cand in candidates or []:
            cn = _norm_col_name(cand)
            if not cn:
                continue
            for c in cols:
                if cn in _norm_col_name(c):
                    return c
        return None

    def list_cols():
        _df = local_vars.get("df")
        return [str(c) for c in getattr(_df, "columns", [])] if _df is not None else []

    local_vars["find_col"] = find_col
    local_vars["list_cols"] = list_cols
    
    # --- 回归模板工具函数（不依赖 statsmodels，用 scipy.stats 实现） ---
    def fit_linear_regression(y, X, feature_names=None):
        """
        拟合线性回归模型，返回系数、p值、R²、置信区间等。
        
        参数:
            y: 目标变量（1D array 或 Series）
            X: 特征矩阵（2D array 或 DataFrame，每列一个特征）
            feature_names: 特征名称列表（可选，用于输出）
        
        返回:
            dict: {
                "coefficients": [系数列表],
                "pvalues": [p值列表],
                "r_squared": R²,
                "adj_r_squared": 调整R²,
                "n": 样本量,
                "summary": "格式化文本摘要",
                "feature_names": [特征名列表]
            }
        """
        # 转为数值（对象/字符串会被 coercion 为 NaN）
        y_ser = pd.Series(y)
        y_num = pd.to_numeric(y_ser, errors="coerce").to_numpy(dtype=float)
        
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            if feature_names is None:
                try:
                    feature_names = [str(c) for c in X_df.columns]
                except Exception:
                    pass
        else:
            X_arr = np.array(X)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            X_df = pd.DataFrame(X_arr)
        
        X_num = X_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        
        # 移除缺失值
        mask = ~(np.isnan(y_num) | np.isnan(X_num).any(axis=1))
        y_clean = y_num[mask]
        X_clean = X_num[mask, :]
        
        if len(y_clean) < 2:
            return {"error": "样本量不足（n<2）或缺失值过多"}
        
        # 添加截距项
        X_with_intercept = np.column_stack([np.ones(len(X_clean)), X_clean])
        
        # OLS 估计：beta = (X'X)^(-1) X'y
        try:
            beta = np.linalg.lstsq(X_with_intercept, y_clean, rcond=None)[0]
        except:
            return {"error": "矩阵不可逆或数值不稳定"}
        
        # 预测值与残差
        y_pred = X_with_intercept @ beta
        residuals = y_clean - y_pred
        
        # R²
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # 调整R²
        n, k = len(y_clean), len(beta) - 1
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1) if n > k + 1 else r_squared
        
        # 标准误与 t 统计量（用于 p 值）
        mse = ss_res / (n - k - 1) if n > k + 1 else ss_res / max(1, n - 1)
        try:
            var_beta = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            se_beta = np.sqrt(np.diag(var_beta))
            t_stats = beta / se_beta
            # 双侧 t 检验 p 值
            pvalues = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))
        except:
            se_beta = np.full(len(beta), np.nan)
            t_stats = np.full(len(beta), np.nan)
            pvalues = np.full(len(beta), np.nan)
        
        # 95% 置信区间
        t_crit = stats.t.ppf(0.975, n - k - 1) if n > k + 1 else 1.96
        ci_lower = beta - t_crit * se_beta
        ci_upper = beta + t_crit * se_beta
        
        # 格式化摘要
        feature_names = feature_names or [f"特征{i+1}" for i in range(k)]
        summary_lines = [f"线性回归结果 (n={n}):", ""]
        summary_lines.append(f"截距: {beta[0]:.4f} (p={pvalues[0]:.4f}, 95%CI: [{ci_lower[0]:.4f}, {ci_upper[0]:.4f}])")
        for i, name in enumerate(feature_names):
            idx = i + 1
            if idx < len(beta):
                summary_lines.append(f"{name}: {beta[idx]:.4f} (p={pvalues[idx]:.4f}, 95%CI: [{ci_lower[idx]:.4f}, {ci_upper[idx]:.4f}])")
        summary_lines.append(f"\nR² = {r_squared:.4f}, 调整R² = {adj_r_squared:.4f}")
        
        return {
            "coefficients": beta.tolist(),
            "pvalues": pvalues.tolist(),
            "r_squared": float(r_squared),
            "adj_r_squared": float(adj_r_squared),
            "n": int(n),
            "summary": "\n".join(summary_lines),
            "feature_names": ["截距"] + feature_names,
            "ci_lower": ci_lower.tolist(),
            "ci_upper": ci_upper.tolist(),
        }
    
    local_vars["fit_linear_regression"] = fit_linear_regression
    local_vars["stats"] = stats  # 也提供 scipy.stats 供其他统计检验用
    local_vars["_session_id"] = session_id  # 传递给图表保存逻辑
    
    plt.clf()
    
    try:
        # 捕获 print 输出
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        redirected_output = StringIO()
        sys.stdout = redirected_output
        
        exec(code_str, {}, local_vars)
        
        sys.stdout = old_stdout
        print_output = redirected_output.getvalue()
        
        text_res = str(local_vars.get('result', "执行成功"))
        
        # 优先使用 print 的内容作为硬结论，如果没有 print，则使用 result 变量
        final_output = print_output if print_output.strip() else text_res
        
        # 图表保存到 out/{session_id}/ 目录（不再返回 base64）
        img_path = None
        if plt.get_fignums():
            # 从调用栈中提取 session_id（通过 inspect 或全局变量传递）
            session_id = local_vars.get("_session_id", "default")
            out_dir = Path("out") / session_id
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成唯一文件名
            timestamp = int(time.time() * 1000)
            filename = f"chart_{timestamp}.png"
            filepath = out_dir / filename
            
            plt.savefig(str(filepath), format='png', bbox_inches='tight', dpi=300)
            plt.close()
            
            # 返回相对路径（前端通过 /out/{session_id}/{filename} 访问）
            img_path = f"{session_id}/{filename}"
        
        new_df = local_vars['df']
        return final_output, img_path, new_df
    except Exception as e:
        import sys
        sys.stdout = sys.__stdout__ # 恢复 stdout 防止崩溃
        return f"Error: {str(e)}", None, df

# --- API 接口 ---

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    session = get_session_data(req.session_id)
    current_df = session['df']
    result = {}

    # 若前端没传 dataContext，则用后端 profile_context 兜底
    data_context = req.dataContext or session.get("profile_context", "")

    # 统一模式名（兼容旧前端：single/expert_mixed）
    mode = _normalize_mode(getattr(req, "mode", "agent_single"))

    # 处理 @ 引用：把被引用列的预览/摘要注入到 data_context
    try:
        at_tokens = _extract_at_mentions(req.message or "")
        at_ctx = _build_at_context(current_df, at_tokens)
        if at_ctx:
            data_context = (data_context or "") + "\n\n[用户引用(@)数据]\n" + at_ctx
    except Exception:
        pass

    # 联网搜索 + 图片 OCR（Ask/Agent 通用；Ask 用于 prompt，Agent 注入 data_context）
    web_results = _duckduckgo_search(req.message, max_results=5) if bool(getattr(req, "webSearch", False)) else []
    web_block = ""
    if web_results:
        lines = []
        for i, r in enumerate(web_results, 1):
            lines.append(f"{i}. {r.get('title','')}\n   {r.get('snippet','')}\n   {r.get('url','')}")
        web_block = "\n\n[联网搜索结果]\n" + "\n".join(lines)

    img_paths = list(getattr(req, "imagePaths", []) or [])
    ocr_obj = _try_ocr_images(img_paths) if img_paths else {}

    # 视觉理解：使用 qwen-omni-turbo 把图片转为"文字描述"，注入到后续 prompt/data_context
    vision_enabled = bool(getattr(req, "visionEnabled", True))
    vision_obj: Dict[str, Any] = {}
    if img_paths and vision_enabled:
        try:
            vision_obj = _vision_understand_images(
                session=session,
                user_query=req.message or "",
                image_paths=img_paths,
                api_keys=req.apiKeys or {},
                vision_enabled=vision_enabled,
                vision_provider="qwen",  # 固定使用 qwen
                vision_model_override="qwen-omni-turbo",  # 固定使用 qwen-omni-turbo
            )
        except Exception as e:
            import traceback
            err_detail = str(e)
            vision_obj = {
                "enabled": vision_enabled,
                "provider": None,
                "model": None,
                "results": [{"path": p, "ok": False, "error": f"视觉理解异常: {err_detail}"} for p in img_paths[:3]],
                "truncated": False,
                "note": f"vision_exception: {err_detail}"
            }

    vision_block = ""
    if img_paths and vision_enabled:
        if isinstance(vision_obj, dict) and vision_obj.get("provider"):
            prov = str(vision_obj.get("provider"))
            model_used = str(vision_obj.get("model") or "")
            provider_label = "Qwen-Omni-Turbo"
            v_lines = [f"\n\n[视觉理解（{provider_label} · {model_used}）]"]
            results = vision_obj.get("results") or []
            has_success = False
            for item in results:
                if not isinstance(item, dict):
                    continue
                p = str(item.get("path") or "")
                if item.get("ok") and item.get("text"):
                    has_success = True
                    t = _truncate_text(item.get("text"), 2500)
                    t = "\n  ".join(str(t).splitlines())
                    v_lines.append(f"- {p}\n  {t}")
                else:
                    err = str(item.get("error") or "视觉理解失败")
                    v_lines.append(f"- {p}\n  ⚠️ {err}")
            if bool(vision_obj.get("truncated")) and len(img_paths) > len(results):
                v_lines.append(f"- （仅处理前 {len(results)} 张图片，其余 {len(img_paths) - len(results)} 张省略）")
            if v_lines:
                vision_block = "\n".join(v_lines)
        elif isinstance(vision_obj, dict) and vision_obj.get("note") == "missing_vision_key":
            vision_block = "\n\n[视觉理解]\n- 未配置千问（Qwen）API Key，无法调用 Qwen-Omni-Turbo（将仅使用 OCR/文件名）。"
        elif isinstance(vision_obj, dict) and vision_obj.get("note") and vision_obj.get("note") not in ("ok", "disabled_or_no_images"):
            note = str(vision_obj.get("note") or "")
            # 如果有错误结果，也显示出来
            results = vision_obj.get("results") or []
            if results:
                v_lines = [f"\n\n[视觉理解]\n- 状态：{note}"]
                for item in results[:3]:
                    if isinstance(item, dict):
                        p = str(item.get("path") or "")
                        err = str(item.get("error") or "视觉理解失败")
                        v_lines.append(f"- {p}\n  ⚠️ {err}")
                vision_block = "\n".join(v_lines)
            else:
                vision_block = f"\n\n[视觉理解]\n- 已跳过：{note}"
    img_block = ""
    if img_paths:
        img_block = "\n\n[图片附件]\n" + "\n".join([f"- {p}" for p in img_paths[:5]])
        if isinstance(ocr_obj, dict) and ocr_obj.get("ocr"):
            for item in ocr_obj.get("ocr", [])[:3]:
                img_block += f"\n\n[OCR提取：{item.get('path')}]\n{item.get('text')}"
        elif isinstance(ocr_obj, dict) and ocr_obj.get("note"):
            img_block += f"\n\n[图片说明]\n{ocr_obj.get('note')}"
        # 视觉理解结果（成功或失败都显示，帮助用户了解状态）
        if vision_block:
            img_block += vision_block
        elif vision_enabled and img_paths:
            # 如果启用了视觉理解但没有生成 vision_block，添加提示
            img_block += "\n\n[视觉理解]\n- 视觉理解未返回结果，请检查千问（Qwen）API Key 配置或网络连接。"

    if mode != "ask":
        extra = (web_block + img_block).strip()
        if extra:
            data_context = (data_context or "") + "\n\n" + extra

    # --- Ask 模式：普通单API对话（可选联网搜索/图片） ---
    if mode == "ask":
        sel = (req.modelSelection or {}) if isinstance(req.modelSelection, dict) else {}
        provider = str(sel.get("provider") or "deepseekA")
        model_name = sel.get("model")
        api_key = (req.apiKeys or {}).get(provider, "")
        if not api_key:
            return {"reply": f"⚠️ Ask 模式未配置 {provider} 的 API Key。请到设置中填写。", "error": True, "process_log": "ask_missing_key"}

        prompt = (
            "你是 Radarm 的 Ask 模式助手（普通问答）。\n"
            "说明：Ask 模式不会自动改动数据；如需对数据进行可回放的清洗/建模，请提示用户切换到 Agent 模式。\n\n"
            f"[用户问题]\n{req.message}\n\n"
            f"[数据概况]\n{data_context}\n"
            f"{web_block}"
            f"{img_block}"
            + ("\n\n【重要提示 - 图片和视觉理解数据】\n"
               "如果上面包含\"[视觉理解]\"或\"[图片附件]\"部分：\n"
               "1. **完整阅读视觉理解结果**：仔细阅读并理解图片中的所有信息，包括但不限于文字、表格、图表、标准、规范、界面元素、图像内容等\n"
               "2. **充分利用图片信息**：根据用户问题和视觉理解结果，提取并使用图片中的任何相关信息来回答用户的问题\n"
               "3. **准确引用**：如果需要在回答中引用图片中的信息（数据、标准、文字、图表内容等），请准确引用视觉理解结果中的内容\n"
               "4. **结构化数据**：如果图片包含表格、标准、规范等结构化信息，且用户需要基于这些信息进行分析或计算，请确保准确提取和使用这些数据\n"
               "5. **完整性检查**：如果视觉理解结果不完整或不确定，请在回答中说明，并基于已有信息尽可能回答用户的问题\n"
               if img_block else "")
        )

        reply = _call_llm(provider, api_key, prompt, model=str(model_name) if model_name else None, temperature=0.3, timeout=180)
        if not reply:
            return {"reply": "⚠️ Ask 模式请求失败或超时。", "error": True, "process_log": "ask_timeout"}
        return {
            "reply": reply,
            "generated_code": None,
            "execution_result": None,
            "image": None,
            "new_data_preview": sanitize_df_for_json(current_df.head(2000)),
            "rows": len(current_df),
            "cols": len(current_df.columns),
            "data_changed": False,
            "history": _history_info(session),
            "history_stack": _history_stack(session),
            "meta": _meta_current(session),
            "profile": session.get("profile", {}),
            "process_log": json.dumps(
                {
                    "type": "ask",
                    "provider": provider,
                    "model": model_name,
                    "web_search": bool(getattr(req, "webSearch", False)),
                    "images": img_paths,
                    "vision": {
                        "enabled": bool(vision_enabled),
                        "provider_pref": str(getattr(req, "visionProvider", "auto") or "auto"),
                        "model_override": getattr(req, "visionModel", None),
                        "provider_used": (vision_obj.get("provider") if isinstance(vision_obj, dict) else None),
                        "model_used": (vision_obj.get("model") if isinstance(vision_obj, dict) else None),
                        "note": (vision_obj.get("note") if isinstance(vision_obj, dict) else None),
                    },
                },
                ensure_ascii=False,
            ),
        }

    # --- 清洗/变换意图：返回 Action 建议（待确认，不直接改数据）---
    clean_keywords = ["缺失", "空值", "删除", "去重", "标准化", "归一化", "替换", "填充", "重命名", "改名", "类型", "转换", "trim", "清洗", "变换", "处理", "999"]
    is_cleaning_intent = any(k in (req.message or "") for k in clean_keywords)
    if is_cleaning_intent:
        # 若用户是“清洗 + 后续分析/建模”的复合任务，则记录 pending，便于用户点“应用所选操作”后自动继续
        analysis_keywords = [
            "分析",
            "统计",
            "描述",
            "频数",
            "列联",
            "交叉",
            "相关",
            "回归",
            "t检验",
            "T检验",
            "方差",
            "ANOVA",
            "卡方",
            "非参",
            "正态",
            "PCA",
            "主成分",
            "聚类",
            "KMeans",
            "kmeans",
            "logistic",
            "线性",
            "模型",
            "预测",
            "显著",
            "图",
            "可视化",
        ]
        has_analysis_intent = any(k in (req.message or "") for k in analysis_keywords)
        if mode in ("agent_single", "agent_multi") and has_analysis_intent:
            session["agent_pending"] = {"kind": "after_apply", "original_message": req.message or "", "created_at": time.time()}
            save_session_to_disk(session)

        # Agent 模式清洗建议默认走“单模型”（可从 modelSelection 取 provider），多专家不在这里做编排
        sel = (req.modelSelection or {}) if isinstance(req.modelSelection, dict) else {}
        clean_provider = str(sel.get("provider") or "deepseekA")
        clean_model = sel.get("model")
        clean_mc = _apply_model_overrides(MODEL_CONFIG, {clean_provider: str(clean_model)} if clean_model else {})
        sug = suggest_clean_actions(
            user_query=req.message or "",
            data_context=data_context,
            api_keys=req.apiKeys,
            primary_model=clean_provider,
            model_config=clean_mc,
            available_columns=[str(c) for c in current_df.columns],
        )
        suggested_actions = sug.get("suggested_actions") or []
        code_preview = actions_to_pandas_code(suggested_actions) if suggested_actions else ""
        process_log_obj = {
            "type": "clean_suggest",
            "provider": sug.get("provider"),
            "suggested_actions": suggested_actions,
            "risk_notes": sug.get("risk_notes", []) or [],
            "note": "清洗建议不会自动改数据，需点击“应用所选操作”后才会生效。",
        }
        return {
            "reply": sug.get("reply", "我给出了一组清洗建议，请确认后应用。"),
            "needs_confirmation": True,
            "suggested_actions": suggested_actions,
            "risk_notes": sug.get("risk_notes", []) or [],
            "generated_code": code_preview or None,
            "execution_result": None,
            "image": None,
            "new_data_preview": sanitize_df_for_json(current_df.head(2000)),
            "rows": len(current_df),
            "cols": len(current_df.columns),
            "data_changed": False,
            "history": _history_info(session),
            "process_log": json.dumps(process_log_obj, ensure_ascii=False, indent=2),
        }

    # --- 快速统计意图（无需调用 LLM，多专家模式也秒回）---
    msg = (req.message or "").strip()
    if msg:
        def _pick_col_from_msg(m: str, df: pd.DataFrame) -> Optional[str]:
            cols = [str(c) for c in getattr(df, "columns", [])]
            hits = [c for c in cols if c and c in m]
            if hits:
                return max(hits, key=len)
            # fallback: 取 “计算XXX均值/平均/中位数/最大/最小/标准差” 的 XXX
            m2 = re.search(r"(?:计算|求)?\s*([^，,。]+?)\s*(?:的)?\s*(均值|平均值|平均|中位数|最大值|最大|最小值|最小|标准差|方差)", m)
            if m2:
                guess = (m2.group(1) or "").strip()
                if guess:
                    # 允许模糊包含匹配
                    for c in cols:
                        if guess and guess in str(c):
                            return str(c)
            return None

        # 是否属于“简单统计”问题（避免误伤复杂任务）
        simple_kw = ["均值", "平均", "中位数", "最大", "最小", "标准差", "方差", "行数", "记录数", "样本量", "列数", "count", "mean", "median", "std", "var"]
        is_simple = any(k in msg for k in simple_kw) and len(msg) <= 40
        if is_simple:
            df0 = current_df
            col = _pick_col_from_msg(msg, df0)
            # 行/列数类无需列名
            if any(k in msg for k in ["行数", "记录数", "样本量", "count"]):
                n_rows = int(len(df0))
                reply = f"**数据行数**：{n_rows}"
                return {
                    "reply": reply,
                    "generated_code": "n_rows = len(df)\nprint(n_rows)\n",
                    "execution_result": str(n_rows),
                    "image": None,
                    "new_data_preview": sanitize_df_for_json(df0.head(2000)),
                    "rows": len(df0),
                    "cols": len(df0.columns),
                    "data_changed": False,
                    "history": _history_info(session),
                    "process_log": "fast_stats=row_count",
                }
            if "列数" in msg:
                n_cols = int(len(df0.columns))
                reply = f"**数据列数**：{n_cols}"
                return {
                    "reply": reply,
                    "generated_code": "n_cols = len(df.columns)\nprint(n_cols)\n",
                    "execution_result": str(n_cols),
                    "image": None,
                    "new_data_preview": sanitize_df_for_json(df0.head(2000)),
                    "rows": len(df0),
                    "cols": len(df0.columns),
                    "data_changed": False,
                    "history": _history_info(session),
                    "process_log": "fast_stats=col_count",
                }

            if col:
                s = pd.to_numeric(df0[col], errors="coerce")
                n = int(s.notna().sum())
                if n == 0:
                    reply = f"⚠️ 列 **{col}** 无有效数值（可能是非数值列或缺失过多）。"
                    return {
                        "reply": reply,
                        "generated_code": f"s = pd.to_numeric(df[{col!r}], errors='coerce')\nprint(int(s.notna().sum()))\n",
                        "execution_result": reply,
                        "image": None,
                        "new_data_preview": sanitize_df_for_json(df0.head(2000)),
                        "rows": len(df0),
                        "cols": len(df0.columns),
                        "data_changed": False,
                        "history": _history_info(session),
                        "process_log": "fast_stats=no_valid_numeric",
                    }

                parts = []
                code_lines = [
                    "import pandas as pd",
                    f"s = pd.to_numeric(df[{col!r}], errors='coerce')",
                    "s = s.dropna()",
                    f"n = int(len(s))",
                ]
                if any(k in msg for k in ["均值", "平均", "平均值", "mean"]):
                    v = float(s.mean())
                    parts.append(f"均值={v:.4f}")
                    code_lines.append("mean_v = float(s.mean())")
                if "中位数" in msg or "median" in msg:
                    v = float(s.median())
                    parts.append(f"中位数={v:.4f}")
                    code_lines.append("median_v = float(s.median())")
                if "最大" in msg:
                    v = float(s.max())
                    parts.append(f"最大值={v:.4f}")
                    code_lines.append("max_v = float(s.max())")
                if "最小" in msg:
                    v = float(s.min())
                    parts.append(f"最小值={v:.4f}")
                    code_lines.append("min_v = float(s.min())")
                if "标准差" in msg or "std" in msg:
                    v = float(s.std())
                    parts.append(f"标准差={v:.4f}")
                    code_lines.append("std_v = float(s.std())")
                if "方差" in msg or "var" in msg:
                    v = float(s.var())
                    parts.append(f"方差={v:.4f}")
                    code_lines.append("var_v = float(s.var())")

                if not parts:
                    v = float(s.mean())
                    parts = [f"均值={v:.4f}"]
                    code_lines.append("mean_v = float(s.mean())")

                reply = f"**{col}**（n={n}）：" + "，".join(parts)
                code_lines.append("print('done')")
                return {
                    "reply": reply,
                    "generated_code": "\n".join(code_lines) + "\n",
                    "execution_result": reply,
                    "image": None,
                    "new_data_preview": sanitize_df_for_json(df0.head(2000)),
                    "rows": len(df0),
                    "cols": len(df0.columns),
                    "data_changed": False,
                    "history": _history_info(session),
                    "process_log": f"fast_stats=col_stats col={col}",
                }

    # --- Agent 模式：SPSS-like 快捷分析路由（优先走确定性 analysis_engine，减少翻车与迭代） ---
    if mode in ("agent_single", "agent_multi"):
        try:
            auto = _maybe_parse_quick_analysis_request(req.message or "", current_df)
        except Exception:
            auto = None

        if isinstance(auto, dict) and auto.get("analysis"):
            analysis_key = str(auto.get("analysis"))
            params = auto.get("params") if isinstance(auto.get("params"), dict) else {}
            title = str(auto.get("title") or analysis_key)

            t0 = time.time()
            try:
                res = run_analysis(session_id=req.session_id, df=current_df, analysis=analysis_key, params=params)
            except Exception:
                res = None

            if isinstance(res, dict):
                tables_obj = res.get("tables") or []
                charts_obj = res.get("charts") or []
                summary_obj = res.get("summary") or {}

                tables = []
                for t in tables_obj:
                    name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else "表格")
                    md = getattr(t, "markdown", None) or (t.get("markdown") if isinstance(t, dict) else "")
                    tables.append({"name": name, "markdown": md})

                charts = []
                for c in charts_obj:
                    name = getattr(c, "name", None) or (c.get("name") if isinstance(c, dict) else "图")
                    path = getattr(c, "path", None) or (c.get("path") if isinstance(c, dict) else None)
                    if path:
                        charts.append({"name": name, "path": path})

                # 解释：用当前模式的“面向用户输出”模型做一次总结（不启用工具调用）
                pref_provider = "deepseekA"
                pref_model = None
                if mode == "agent_multi":
                    roles_in = req.agentRoles if isinstance(req.agentRoles, dict) else {}
                    exec_cfg = roles_in.get("executor") if isinstance(roles_in.get("executor"), dict) else {}
                    pref_provider = str(exec_cfg.get("provider") or "deepseekB")
                    pref_model = exec_cfg.get("model")
                else:
                    sel = (req.modelSelection or {}) if isinstance(req.modelSelection, dict) else {}
                    pref_provider = str(sel.get("provider") or "deepseekA")
                    pref_model = sel.get("model")

                provider = _choose_provider(req.apiKeys or {}, pref_provider or "deepseekA") or "deepseekA"
                api_key = (req.apiKeys or {}).get(provider, "")

                explanation = ""
                if api_key:
                    meta_cur = _meta_current(session)
                    meta_hint = {}
                    try:
                        cols_meta = (meta_cur.get("columns") or {})
                        for k, v in list(cols_meta.items())[:30]:
                            meta_hint[k] = {kk: vv for kk, vv in (v or {}).items() if kk in ("label", "measure")}
                    except Exception:
                        meta_hint = {}

                    def _truncate(s: str, n: int = 3500) -> str:
                        s2 = str(s or "")
                        return s2 if len(s2) <= n else s2[:n] + "\n...(截断)"

                    tables_md = "\n\n".join([f"### {t['name']}\n{t['markdown']}" for t in tables[:3]])
                    prompt = (
                        "你是一名严谨的统计分析助手。请基于下面给定的分析结果，输出一段中文解释（Markdown）。\n"
                        "要求：\n"
                        "- 不要杜撰任何数字；所有数字必须来自给定结果。\n"
                        "- 先给一句话结论，然后列出关键证据（统计量/df/p 值/样本量/效应量如有）。\n"
                        "- 若不显著或数据不足，明确说明，并给出下一步建议。\n"
                        "- 语言清晰、可审计。\n\n"
                        f"[分析类型]\n{title}\n\n"
                        f"[参数(JSON)]\n{_truncate(json.dumps(params or {}, ensure_ascii=False, indent=2), 1200)}\n\n"
                        f"[数据规模]\n{len(current_df)} 行, {len(current_df.columns)} 列\n\n"
                        f"[变量元数据(截断)]\n{_truncate(json.dumps(meta_hint, ensure_ascii=False, indent=2), 1200)}\n\n"
                        f"[结果摘要(JSON)]\n{_truncate(json.dumps(summary_obj, ensure_ascii=False, indent=2), 2000)}\n\n"
                        f"[结果表格(截断)]\n{_truncate(tables_md, 4000)}\n"
                    )
                    explanation = _call_llm(
                        provider,
                        api_key,
                        prompt,
                        model=str(pref_model) if pref_model else None,
                        temperature=0.2,
                        timeout=180,
                    ) or ""

                elapsed_ms = int((time.time() - t0) * 1000)
                return {
                    "title": title,
                    "reply": explanation or f"✅ 已完成：{title}（模型未返回解释）",
                    "tables": tables,
                    "charts": charts,
                    "image": charts[0]["path"] if charts else None,
                    "generated_code": None,
                    "execution_result": None,
                    "new_data_preview": sanitize_df_for_json(current_df.head(2000)),
                    "rows": len(current_df),
                    "cols": len(current_df.columns),
                    "data_changed": False,
                    "history": _history_info(session),
                    "history_stack": _history_stack(session),
                    "meta": _meta_current(session),
                    "profile": session.get("profile", {}),
                    "process_log": json.dumps(
                        {"type": "chat_quick_analysis", "analysis": analysis_key, "params": params, "elapsed_ms": elapsed_ms},
                        ensure_ascii=False,
                    ),
                    "elapsed_ms": elapsed_ms,
                }

    # 包装 execute_code，绑定 session_id
    def execute_with_session(code_str, df):
        return execute_code(code_str, df, session_id=req.session_id)
    
    if mode == "agent_multi":
        # 角色配置：允许前端指定每个角色的 provider+model
        roles_in = req.agentRoles if isinstance(req.agentRoles, dict) else {}
        roles = {"planner": "deepseekA", "executor": "deepseekB", "verifier": "deepseekC"}
        overrides: Dict[str, str] = {}
        for r in ("planner", "executor", "verifier"):
            rc = roles_in.get(r) if isinstance(roles_in.get(r), dict) else {}
            prov = str(rc.get("provider") or roles[r])
            roles[r] = prov
            if rc.get("model"):
                overrides[prov] = str(rc.get("model"))
        mc = _apply_model_overrides(MODEL_CONFIG, overrides)
        result = workflow_multi_chat.run_workflow(
            user_query=req.message,
            data_context=data_context,
            api_keys=req.apiKeys,
            model_config=mc,
            roles=roles,
            execute_callback=execute_with_session,
            df=current_df
        )
    else:
        # agent_single（默认）
        sel = (req.modelSelection or {}) if isinstance(req.modelSelection, dict) else {}
        provider = str(sel.get("provider") or "deepseekA")
        overrides = {provider: str(sel.get("model"))} if sel.get("model") else {}
        mc = _apply_model_overrides(MODEL_CONFIG, overrides)
        result = workflow_single_chat.run_workflow(
            user_query=req.message,
            data_context=data_context,
            api_keys=req.apiKeys,
            primary_model=provider,
            model_config=mc,
            execute_callback=execute_with_session,
            df=current_df
        )

    if "error" in result:
        return {"reply": f"⚠️ {result['error']}", "error": True, "process_log": result.get("process_log", "")}

    data_changed = False
    new_df = result.get("new_df")
    if new_df is not None:
        if len(new_df) != len(current_df) or len(new_df.columns) != len(current_df.columns) or not new_df.equals(current_df):
            update_session_data(req.session_id, new_df)
            data_changed = True
            session = get_session_data(req.session_id)
            current_df = session["df"]

    return {
        "reply": result["reply"],
        "generated_code": result.get("generated_code"),
        "execution_result": result.get("execution_result"),
        "image": result.get("image"),
        "new_data_preview": sanitize_df_for_json(current_df.head(2000)),
        "rows": len(current_df),
        "cols": len(current_df.columns),
        "data_changed": data_changed,
        "history": _history_info(session),
        "history_stack": _history_stack(session),
        "meta": _meta_current(session),
        "profile": session.get("profile", {}),
        "process_log": result.get("process_log", "")
    }

@app.post("/report")
async def generate_report(req: ReportRequest):
    """
    一键生成数据分析报告（Report v2）：多模型工作流 + 图表 + 图表数据（CSV）
    """
    session = get_session_data(req.session_id)
    _ensure_reports(session)
    df = session["df"]

    # 预生成 report_id，便于异步返回
    report_id = None
    if (not req.saveAsNew) and req.reportId:
        report_id = _safe_id(req.reportId)
    else:
        report_id = _safe_id(uuid.uuid4().hex)

    if getattr(req, "runAsync", False):
        task = celery_app.send_task(
            "radarm.report",
            kwargs={
                "session_id": req.session_id,
                "req_obj": req.dict(),
                "report_id": report_id,
            },
        )
        return {"task_id": task.id, "report_id": report_id, "state": "queued", "message": "报告生成已提交"}

    # 选取部分数据：列选择 + @ 引用列
    selected_cols: List[str] = []
    try:
        if isinstance(req.selectedColumns, list) and req.selectedColumns:
            selected_cols.extend([str(c) for c in req.selectedColumns if c is not None])
    except Exception:
        selected_cols = []

    try:
        at_tokens = _extract_at_mentions(req.userRequest or "")
        cols_all = [str(c) for c in getattr(df, "columns", [])]
        for t in at_tokens:
            col = _resolve_mention_to_column(t, cols_all)
            if col:
                selected_cols.append(col)
    except Exception:
        pass

    # 去重保序，过滤不存在列
    seen = set()
    cols_exist = [str(c) for c in getattr(df, "columns", [])]
    filtered_cols: List[str] = []
    for c in selected_cols:
        c = str(c)
        if c in cols_exist and c not in seen:
            filtered_cols.append(c)
            seen.add(c)

    df_use = df
    if filtered_cols:
        try:
            df_use = df[filtered_cols].copy()
        except Exception:
            df_use = df

    # 抽样（可选）
    sample_rows = None
    try:
        if req.sampleRows is not None:
            n = int(req.sampleRows)
            if n > 0:
                sample_rows = n
                if len(df_use) > n:
                    df_use = df_use.sample(n=n, random_state=42)
    except Exception:
        sample_rows = None

    # 数据上下文（基于 df_use）
    try:
        desc = df_use.describe().to_markdown()
    except Exception:
        try:
            desc = str(df_use.describe())
        except Exception:
            desc = ""

    data_shape = f"{len(df_use)} 行, {len(df_use.columns)} 列"
    cols_preview = ", ".join([str(c) for c in df_use.columns[:80]])
    if len(df_use.columns) > 80:
        cols_preview += f" ...（其余 {len(df_use.columns) - 80} 列省略）"

    try:
        missing_pct = (df_use.isna().mean() * 100).sort_values(ascending=False)
        missing_top = missing_pct.head(15).to_dict()
    except Exception:
        missing_top = {}

    sample_rows_preview = str(sanitize_df_for_json(df_use.head(5)))

    data_context = (
        f"来源: {session.get('filename','data.csv')}\n"
        f"规模: {data_shape}\n"
        f"字段(预览): {cols_preview}\n"
        f"缺失率Top15(%): {missing_top}\n"
        f"统计摘要(数值列):\n{desc}\n"
        f"样例行(前5行): {sample_rows_preview}"
    )

    # 报告 ID（多份报告）已在前面生成

    # 设置生成跟踪标志
    if req.session_id not in report_generation_tracking:
        report_generation_tracking[req.session_id] = {}
    report_generation_tracking[req.session_id][report_id] = True

    # 定义取消检查函数
    def check_cancelled() -> bool:
        """检查是否已取消生成"""
        return not report_generation_tracking.get(req.session_id, {}).get(report_id, False)

    # 执行报告工作流
    try:
        result = run_report_engine_v2(
            session_id=req.session_id,
            report_id=report_id,
            user_request=req.userRequest or "",
            data_context=data_context,
            api_keys=req.apiKeys or {},
            model_config=MODEL_CONFIG,
            df=df_use,
            stage_models=req.reportStages or {},
            selected_columns=filtered_cols,
            sample_rows=sample_rows,
            check_cancelled=check_cancelled,
        )
    except Exception as e:
        # 清理跟踪标志
        if req.session_id in report_generation_tracking:
            report_generation_tracking[req.session_id].pop(report_id, None)
        raise e

    # 清理跟踪标志
    if req.session_id in report_generation_tracking:
        report_generation_tracking[req.session_id].pop(report_id, None)

    # 检查是否被取消
    if result.get("cancelled"):
        return {"error": "报告生成已取消", "cancelled": True, "process_log": result.get("log", ""), "report_id": report_id}

    if "error" in result:
        return {"error": result["error"], "process_log": result.get("log", ""), "report_id": report_id}

    artifacts = result.get("artifacts") or {}
    charts = artifacts.get("charts") or []
    base_dir = artifacts.get("base_dir")

    # 写入 session.reports 元数据
    entry = {
        "report_id": report_id,
        "title": result.get("title") or "数据分析报告",
        "created_at": time.time(),
        "user_request": req.userRequest or "",
        "selected_columns": filtered_cols,
        "sample_rows": sample_rows,
        "base_dir": base_dir,
        "report_path": artifacts.get("report_path"),
        "manifest_path": artifacts.get("manifest_path"),
        "charts": charts,
    }

    reports = _ensure_reports(session)
    if (not req.saveAsNew) and req.reportId:
        replaced = False
        for i, r in enumerate(reports):
            if isinstance(r, dict) and str(r.get("report_id")) == str(report_id):
                reports[i] = entry
                replaced = True
                break
        if not replaced:
            reports.append(entry)
    else:
        reports.append(entry)

    session["reports"] = reports
    save_session_to_disk(session)
    sessions[req.session_id] = session

    return {
        "report_id": report_id,
        "title": result.get("title") or "数据分析报告",
        "report": result.get("content") or "",
        "charts": charts,
        "tables": (artifacts.get("tables") or []),
        "process_log": result.get("log") or "",
        "reports": [_report_meta_from_entry(r) for r in reports if isinstance(r, dict)],
    }


@app.post("/report_cancel")
async def cancel_report_generation(req: ReportCancelRequest):
    """
    取消正在生成的报告
    如果 report_id 为 None，则取消该 session 的所有正在生成的报告
    """
    session_id = req.session_id
    report_id = req.report_id
    
    if session_id not in report_generation_tracking:
        return {"message": "没有正在生成的报告", "cancelled": False}
    
    if report_id:
        # 取消特定报告
        report_id = _safe_id(report_id)
        if report_id in report_generation_tracking[session_id]:
            report_generation_tracking[session_id][report_id] = False
            return {"message": f"已取消报告 {report_id} 的生成", "cancelled": True}
        else:
            return {"message": "报告未在生成中", "cancelled": False}
    else:
        # 取消该 session 的所有报告
        count = len(report_generation_tracking[session_id])
        for rid in list(report_generation_tracking[session_id].keys()):
            report_generation_tracking[session_id][rid] = False
        return {"message": f"已取消 {count} 个正在生成的报告", "cancelled": True, "count": count}


@app.get("/report_list")
async def report_list(session_id: str):
    session = get_session_data(session_id)
    reports = _ensure_reports(session)
    return {"session_id": session_id, "reports": [_report_meta_from_entry(r) for r in reports if isinstance(r, dict)]}


@app.get("/report_get")
async def report_get(session_id: str, report_id: str):
    session = get_session_data(session_id)
    reports = _ensure_reports(session)
    rid = _safe_id(report_id)

    meta = None
    for r in reports:
        if isinstance(r, dict) and str(r.get("report_id")) == str(rid):
            meta = r
            break
    if not meta:
        raise HTTPException(status_code=404, detail="report_id 不存在")

    rdir = _report_dir(session_id, rid)
    md_path = rdir / "report.md"
    manifest_path = rdir / "manifest.json"
    if not md_path.exists():
        raise HTTPException(status_code=404, detail="报告文件不存在（可能已被清理）")

    content = md_path.read_text(encoding="utf-8", errors="ignore")
    manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = None

    return {
        "report_id": rid,
        "title": meta.get("title") or "数据分析报告",
        "report": content,
        "charts": meta.get("charts") or [],
        "meta": _report_meta_from_entry(meta),
        "manifest": manifest,
        "process_log": (manifest.get("process_log") if isinstance(manifest, dict) else ""),
    }


@app.get("/download_report_bundle")
async def download_report_bundle(session_id: str, report_id: str):
    """
    导出 ZIP：report.md + manifest.json + charts + csv（图表数据）
    """
    rid = _safe_id(report_id)
    rdir = _report_dir(session_id, rid)
    if not rdir.exists():
        raise HTTPException(status_code=404, detail="报告目录不存在（可能已被清理）")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in rdir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(rdir)
                zf.write(p, arcname=str(rel).replace("\\", "/"))
    buf.seek(0)
    fname = urllib.parse.quote(f"report_{rid}.zip")
    return StreamingResponse(buf, media_type="application/zip", headers={"Content-Disposition": f"attachment; filename={fname}"})
    
    # 准备数据上下文（尽量提供“原始片段/字段画像/质量线索”，便于 Stage1 分诊与 Stage2 写代码）
    try:
        desc = df.describe().to_markdown()
    except Exception:
        try:
            desc = str(df.describe())
        except Exception:
            desc = ""

    data_shape = f"{len(df)} 行, {len(df.columns)} 列"
    cols_preview = ", ".join([str(c) for c in df.columns[:60]])
    if len(df.columns) > 60:
        cols_preview += f" ...（其余 {len(df.columns) - 60} 列省略）"

    # 缺失率 TopN
    try:
        missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
        missing_top = missing_pct.head(15).to_dict()
    except Exception:
        missing_top = {}

    # 字段画像（截断列数，避免 prompt 过大）
    schema_lines = []
    max_cols_profile = 50
    for col in list(df.columns)[:max_cols_profile]:
        try:
            s = df[col]
            dtype = str(s.dtype)
            miss = float(s.isna().mean() * 100)
            nunique = int(s.nunique(dropna=True))
            examples = s.dropna().astype(str).unique()[:3].tolist()
            examples = [e[:30] for e in examples]
            schema_lines.append(f"- {col} ({dtype}), missing={miss:.1f}%, unique≈{nunique}, examples={examples}")
        except Exception:
            schema_lines.append(f"- {col} (unknown)")
    if len(df.columns) > max_cols_profile:
        schema_lines.append(f"...（其余 {len(df.columns) - max_cols_profile} 列省略）")
    schema_profile = "\n".join(schema_lines)

    sample_rows = str(sanitize_df_for_json(df.head(5)))

    data_context = (
        f"来源: {session['filename']}\n"
        f"规模: {data_shape}\n"
        f"字段(预览): {cols_preview}\n"
        f"缺失率Top15(%): {missing_top}\n"
        f"字段画像:\n{schema_profile}\n"
        f"统计摘要(数值列):\n{desc}\n"
        f"样例行(前5行): {sample_rows}"
    )

    # 包装 execute_code，绑定 session_id
    def execute_with_session(code_str, df):
        return execute_code(code_str, df, session_id=req.session_id)

    # 调用新的五阶段工作流
    # 注意：这里我们传入了 execute_code 回调函数，让 Stage 2 可以跑代码
    result = workflow_report.run_workflow(
        user_request=req.userRequest,
        data_context=data_context,
        api_keys=req.apiKeys,
        model_config=MODEL_CONFIG,
        execute_callback=execute_with_session,
        df=df
    )
    
    if "error" in result:
        return {"error": result["error"], "process_log": result.get("log", "")}
    return {"report": result["content"], "process_log": result["log"]}

# --- 基础接口 ---
@app.post("/upload")
async def upload_file(session_id: str = Form(...), file: UploadFile = File(...)):
    try:
        contents = await file.read()
        buffer = io.BytesIO(contents)
        if file.filename.endswith('.csv'):
            try: df = pd.read_csv(buffer, encoding='utf-8')
            except: 
                try: df = pd.read_csv(buffer, encoding='gbk')
                except: df = pd.read_csv(buffer, encoding='gb18030')
        elif file.filename.endswith(('.xls', '.xlsx')): df = pd.read_excel(buffer)
        elif file.filename.endswith('.json'): df = pd.read_json(buffer)
        elif file.filename.endswith('.parquet'): df = pd.read_parquet(buffer)
        else: raise HTTPException(status_code=400, detail="不支持格式")
        df = df.replace([float('inf'), float('-inf')], float('nan'))
        update_session_data(session_id, df, file.filename)
        session = get_session_data(session_id)
        return {
            "session_id": session_id,
            "filename": file.filename,
            "preview": sanitize_df_for_json(session["df"].head(2000)),
            "rows": len(session["df"]),
            "cols": len(session["df"].columns),
            "history": _history_info(session),
            "history_stack": _history_stack(session),
            "meta": _meta_current(session),
            "profile": session.get("profile", {}),
        }
    except Exception as e: raise HTTPException(status_code=500, detail=f"解析失败: {str(e)}")

@app.post("/connect_db")
async def connect_database(req: DBConnectRequest):
    try:
        db_url = ""
        if req.type == 'mysql': db_url = f"mysql+pymysql://{req.user}:{req.password}@{req.host}:{req.port}/{req.database}"
        elif req.type == 'postgres': db_url = f"postgresql+psycopg2://{req.user}:{req.password}@{req.host}:{req.port}/{req.database}"
        elif req.type == 'sqlite':
            db_path = req.database.strip().replace('\\', '/')
            if not os.path.exists(db_path): raise HTTPException(status_code=400, detail=f"文件不存在")
            db_url = f"sqlite:///{db_path}" 
        else: raise HTTPException(status_code=400, detail="不支持数据库")
        engine = create_engine(db_url)
        with engine.connect() as conn: df = pd.read_sql(req.sql, conn)
        df = df.replace([float('inf'), float('-inf')], float('nan'))
        update_session_data(req.session_id, df, f"{req.type}_result")
        session = get_session_data(req.session_id)
        return {
            "session_id": req.session_id,
            "filename": f"DB: {req.database}",
            "preview": sanitize_df_for_json(session["df"].head(2000)),
            "rows": len(session["df"]),
            "cols": len(session["df"].columns),
            "history": _history_info(session),
            "history_stack": _history_stack(session),
            "meta": _meta_current(session),
            "profile": session.get("profile", {}),
        }
    except Exception as e: raise HTTPException(status_code=500, detail=f"连接失败: {str(e)}")

@app.get("/download")
async def download_data(session_id: str):
    session = get_session_data(session_id)
    df = session['df']
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    fname = urllib.parse.quote(f"processed_{session['filename']}.csv")
    response.headers["Content-Disposition"] = f"attachment; filename={fname}"
    return response

@app.post("/download_report")
async def download_report_endpoint(req: ReportDownloadRequest):
    stream = io.StringIO(req.content)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/markdown")
    fname = urllib.parse.quote(f"{req.filename}.md")
    response.headers["Content-Disposition"] = f"attachment; filename={fname}"
    return response

@app.post("/reset")
async def reset_project(req: ResetRequest):
    session = get_session_data(req.session_id)
    session["raw_df"] = _placeholder_df()
    session["actions"] = []
    session["cursor"] = 0
    session["filename"] = "data.csv"
    session["meta"] = {"columns": {}}
    session["reports"] = []
    session["agent_pending"] = None
    _ensure_meta(session)
    _recompute_current_df(session)
    save_session_to_disk(session)
    sessions[req.session_id] = session
    
    # 清理该任务的 out 目录
    out_dir = Path("out") / req.session_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    
    return {
        "message": "已重置",
        "filename": session.get("filename", "data.csv"),
        "history": _history_info(session),
        "history_stack": _history_stack(session),
        "meta": _meta_current(session),
        "preview": sanitize_df_for_json(session["df"].head(2000)),
        "rows": len(session["df"]),
        "cols": len(session["df"].columns),
    }


@app.get("/session_state")
async def session_state(session_id: str):
    session = get_session_data(session_id)
    df = session["df"]
    code = actions_to_pandas_code((session.get("actions") or [])[: int(session.get("cursor", 0))])
    return {
        "session_id": session_id,
        "filename": session.get("filename", "data.csv"),
        "preview": sanitize_df_for_json(df.head(2000)),
        "rows": len(df),
        "cols": len(df.columns),
        "history": _history_info(session),
        "history_stack": _history_stack(session),
        "meta": _meta_current(session),
        "profile": session.get("profile", {}),
        "generated_code": code if code.strip() else None,
    }


@app.post("/apply_actions")
async def apply_actions_endpoint(req: ApplyActionsRequest):
    session = get_session_data(req.session_id)
    ok, err = validate_actions(req.actions)
    if not ok:
        raise HTTPException(status_code=400, detail=err)

    # 分支：若 cursor 不在末尾，则截断 redo 栈
    actions_all = (session.get("actions") or [])[: int(session.get("cursor", 0))] + (req.actions or [])
    cursor = len(actions_all)

    try:
        new_df = apply_actions(session["raw_df"], actions_all)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"应用失败: {str(e)}")

    session["actions"] = actions_all
    session["cursor"] = cursor
    session["df"] = new_df
    prof = build_data_profile(new_df)
    session["profile"] = prof
    session["profile_context"] = profile_to_context(prof)
    # 应用动作后，可能需要继续执行“Agent 后续任务”（例如：清洗后继续分析）
    agent_followup = None
    save_session_to_disk(session)

    code = actions_to_pandas_code(actions_all)

    # 若携带 pending 且允许自动续跑，则调用 tool-first agent（不再需要用户再发一条消息）
    try:
        pending = session.get("agent_pending")
        mode2 = _normalize_mode(getattr(req, "mode", "agent_single"))
        if req.autoContinue and isinstance(pending, dict) and pending.get("kind") == "after_apply":
            # 必须有 key 才能续跑
            api_keys = req.apiKeys or {}
            if isinstance(api_keys, dict) and any(v for v in api_keys.values()):
                # 续跑 query：提示“清洗已完成”，避免重复建议清洗
                original = str(pending.get("original_message") or "").strip()
                briefs = []
                try:
                    for a in (req.actions or [])[:8]:
                        briefs.append(action_to_brief(a))
                except Exception:
                    briefs = []
                brief_text = "；".join(briefs) if briefs else f"{len(req.actions or [])} 个清洗/变换操作"

                follow_query = (
                    "你正在继续完成一个多步任务：\n"
                    f"- 原始用户任务：{original}\n"
                    f"- 已执行清洗/变换：{brief_text}\n\n"
                    "请注意：清洗已完成，请不要再输出清洗建议卡片，直接继续完成剩余分析/建模/可视化任务。\n"
                    "重要约束：除非用户明确要求“把变换后的数据写回并作为后续对话基线”，否则不要修改 df（不要对 df 重新赋值/新增列写回）。如需派生变量用于分析，请使用局部变量或 df_tmp=df.copy()，不要写回 df。\n"
                )

                # 绑定 session_id 的沙盒执行
                def execute_with_session(code_str, df0):
                    return execute_code(code_str, df0, session_id=req.session_id)

                follow_result = None
                # 优先：若原始用户任务是明确的“SPSS 风格分析”，直接走确定性分析引擎（更快、更稳）
                try:
                    auto2 = _maybe_parse_quick_analysis_request(original, new_df)
                except Exception:
                    auto2 = None

                if isinstance(auto2, dict) and auto2.get("analysis"):
                    analysis_key = str(auto2.get("analysis"))
                    params = auto2.get("params") if isinstance(auto2.get("params"), dict) else {}
                    title = str(auto2.get("title") or analysis_key)

                    t0 = time.time()
                    try:
                        res = run_analysis(session_id=req.session_id, df=new_df, analysis=analysis_key, params=params)
                    except Exception:
                        res = None

                    if isinstance(res, dict):
                        tables_obj = res.get("tables") or []
                        charts_obj = res.get("charts") or []
                        summary_obj = res.get("summary") or {}

                        tables = []
                        for t in tables_obj:
                            name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else "表格")
                            md = getattr(t, "markdown", None) or (t.get("markdown") if isinstance(t, dict) else "")
                            tables.append({"name": name, "markdown": md})

                        images = []
                        for c in charts_obj:
                            path = getattr(c, "path", None) or (c.get("path") if isinstance(c, dict) else None)
                            if path:
                                images.append(path)

                        # 解释：按当前模式的模型偏好（单模型/多专家 executor）做一次结果总结
                        pref_provider = "deepseekA"
                        pref_model = None
                        if mode2 == "agent_multi":
                            roles_in = req.agentRoles if isinstance(req.agentRoles, dict) else {}
                            exec_cfg = roles_in.get("executor") if isinstance(roles_in.get("executor"), dict) else {}
                            pref_provider = str(exec_cfg.get("provider") or "deepseekB")
                            pref_model = exec_cfg.get("model")
                        else:
                            sel = req.modelSelection if isinstance(req.modelSelection, dict) else {}
                            pref_provider = str(sel.get("provider") or "deepseekA")
                            pref_model = sel.get("model")

                        provider = _choose_provider(api_keys, pref_provider or "deepseekA") or "deepseekA"
                        api_key = (api_keys or {}).get(provider, "")
                        explanation = ""
                        if api_key:
                            meta_cur = _meta_current(session)
                            meta_hint = {}
                            try:
                                cols_meta = (meta_cur.get("columns") or {})
                                for k, v in list(cols_meta.items())[:30]:
                                    meta_hint[k] = {kk: vv for kk, vv in (v or {}).items() if kk in ("label", "measure")}
                            except Exception:
                                meta_hint = {}

                            def _truncate(s: str, n: int = 3500) -> str:
                                s2 = str(s or "")
                                return s2 if len(s2) <= n else s2[:n] + "\n...(截断)"

                            tables_md = "\n\n".join([f"### {t['name']}\n{t['markdown']}" for t in tables[:3]])
                            prompt = (
                                "你是一名严谨的统计分析助手。请基于下面给定的分析结果，输出一段中文解释（Markdown）。\n"
                                "要求：\n"
                                "- 不要杜撰任何数字；所有数字必须来自给定结果。\n"
                                "- 先给一句话结论，然后列出关键证据（统计量/df/p 值/样本量/效应量如有）。\n"
                                "- 若不显著或数据不足，明确说明，并给出下一步建议。\n"
                                "- 语言清晰、可审计。\n\n"
                                f"[分析类型]\n{title}\n\n"
                                f"[参数(JSON)]\n{_truncate(json.dumps(params or {}, ensure_ascii=False, indent=2), 1200)}\n\n"
                                f"[数据规模]\n{len(new_df)} 行, {len(new_df.columns)} 列\n\n"
                                f"[变量元数据(截断)]\n{_truncate(json.dumps(meta_hint, ensure_ascii=False, indent=2), 1200)}\n\n"
                                f"[结果摘要(JSON)]\n{_truncate(json.dumps(summary_obj, ensure_ascii=False, indent=2), 2000)}\n\n"
                                f"[结果表格(截断)]\n{_truncate(tables_md, 4000)}\n"
                            )
                            explanation = _call_llm(
                                provider,
                                api_key,
                                prompt,
                                model=str(pref_model) if pref_model else None,
                                temperature=0.2,
                                timeout=180,
                            ) or ""

                        elapsed_ms = int((time.time() - t0) * 1000)
                        follow_result = {
                            "reply": explanation or f"✅ 已完成：{title}（模型未返回解释）",
                            "tables": tables,
                            "images": images,
                            "image": images[0] if images else None,
                            "process_log": json.dumps(
                                {"type": "after_apply_quick_analysis", "analysis": analysis_key, "params": params, "elapsed_ms": elapsed_ms},
                                ensure_ascii=False,
                            ),
                        }

                if follow_result is None and mode2 == "agent_multi":
                    roles_in = req.agentRoles if isinstance(req.agentRoles, dict) else {}
                    roles = {"planner": "deepseekA", "executor": "deepseekB", "verifier": "deepseekC"}
                    overrides: Dict[str, str] = {}
                    for r in ("planner", "executor", "verifier"):
                        rc = roles_in.get(r) if isinstance(roles_in.get(r), dict) else {}
                        prov = str(rc.get("provider") or roles[r])
                        roles[r] = prov
                        if rc.get("model"):
                            overrides[prov] = str(rc.get("model"))
                    mc = _apply_model_overrides(MODEL_CONFIG, overrides)
                    follow_result = workflow_multi_chat.run_workflow(
                        user_query=follow_query,
                        data_context=session.get("profile_context", ""),
                        api_keys=api_keys,
                        model_config=mc,
                        roles=roles,
                        execute_callback=execute_with_session,
                        df=new_df,
                    )
                elif follow_result is None:
                    sel = req.modelSelection if isinstance(req.modelSelection, dict) else {}
                    provider = str(sel.get("provider") or "deepseekA")
                    overrides = {provider: str(sel.get("model"))} if sel.get("model") else {}
                    mc = _apply_model_overrides(MODEL_CONFIG, overrides)
                    follow_result = workflow_single_chat.run_workflow(
                        user_query=follow_query,
                        api_keys=api_keys,
                        primary_model=provider,
                        model_config=mc,
                        execute_callback=execute_with_session,
                        df=new_df,
                    )

                if isinstance(follow_result, dict) and follow_result.get("reply"):
                    agent_followup = {
                        "reply": follow_result.get("reply"),
                        "generated_code": follow_result.get("generated_code"),
                        "execution_result": follow_result.get("execution_result"),
                        "image": follow_result.get("image"),
                        "tables": follow_result.get("tables") or [],
                        "images": follow_result.get("images") or [],
                        "process_log": follow_result.get("process_log") or "",
                    }

            # 清空 pending（无论是否成功，避免卡死在 pending）
            session["agent_pending"] = None
            save_session_to_disk(session)
    except Exception:
        # 不影响 apply_actions 主流程
        try:
            session["agent_pending"] = None
            save_session_to_disk(session)
        except Exception:
            pass

    return {
        "message": f"✅ 已应用 {len(req.actions or [])} 个操作（当前共 {cursor} 步）。",
        "applied_actions": req.actions or [],
        "generated_code": code,
        "new_data_preview": sanitize_df_for_json(new_df.head(2000)),
        "rows": len(new_df),
        "cols": len(new_df.columns),
        "history": _history_info(session),
        "history_stack": _history_stack(session),
        "meta": _meta_current(session),
        "profile": prof,
        "data_changed": True,
        "agent_followup": agent_followup,
    }


@app.post("/undo")
async def undo_endpoint(req: ResetRequest):
    session = get_session_data(req.session_id)
    cursor = int(session.get("cursor", 0))
    if cursor <= 0:
        return {"message": "已经在最初状态，无法撤销。", "history": _history_info(session)}
    session["cursor"] = cursor - 1
    _recompute_current_df(session)
    save_session_to_disk(session)
    df = session["df"]
    code = actions_to_pandas_code((session.get("actions") or [])[: int(session.get("cursor", 0))])
    return {
        "message": "↩️ 已撤销一步。",
        "generated_code": code if code.strip() else None,
        "new_data_preview": sanitize_df_for_json(df.head(2000)),
        "rows": len(df),
        "cols": len(df.columns),
        "history": _history_info(session),
        "history_stack": _history_stack(session),
        "meta": _meta_current(session),
        "profile": session.get("profile", {}),
        "data_changed": True,
    }


@app.post("/redo")
async def redo_endpoint(req: ResetRequest):
    session = get_session_data(req.session_id)
    cursor = int(session.get("cursor", 0))
    total = len(session.get("actions") or [])
    if cursor >= total:
        return {"message": "已经在最新状态，无法重做。", "history": _history_info(session)}
    session["cursor"] = cursor + 1
    _recompute_current_df(session)
    save_session_to_disk(session)
    df = session["df"]
    code = actions_to_pandas_code((session.get("actions") or [])[: int(session.get("cursor", 0))])
    return {
        "message": "↪️ 已重做一步。",
        "generated_code": code if code.strip() else None,
        "new_data_preview": sanitize_df_for_json(df.head(2000)),
        "rows": len(df),
        "cols": len(df.columns),
        "history": _history_info(session),
        "history_stack": _history_stack(session),
        "meta": _meta_current(session),
        "profile": session.get("profile", {}),
        "data_changed": True,
    }


@app.post("/set_cursor")
async def set_cursor_endpoint(req: SetCursorRequest):
    session = get_session_data(req.session_id)
    total = len(session.get("actions") or [])
    cursor = int(req.cursor)
    cursor = max(0, min(cursor, total))
    session["cursor"] = cursor
    _recompute_current_df(session)
    save_session_to_disk(session)
    df = session["df"]
    code = actions_to_pandas_code((session.get("actions") or [])[: int(session.get("cursor", 0))])
    return {
        "message": f"已跳转到第 {cursor}/{total} 步",
        "generated_code": code if code.strip() else None,
        "new_data_preview": sanitize_df_for_json(df.head(2000)),
        "rows": len(df),
        "cols": len(df.columns),
        "history": _history_info(session),
        "history_stack": _history_stack(session),
        "meta": _meta_current(session),
        "profile": session.get("profile", {}),
        "data_changed": True,
    }


@app.post("/update_column_meta")
async def update_column_meta_endpoint(req: ColumnMetaRequest):
    session = get_session_data(req.session_id)
    meta = _ensure_meta(session)
    cols_meta: Dict[str, Any] = meta.get("columns", {})
    maps = _compute_column_maps(session)
    cur_to_orig = maps.get("cur_to_orig", {})

    cur_col = str(req.column)
    orig_col = cur_to_orig.get(cur_col, cur_col)

    measure = (req.measure or "scale").lower().strip()
    if measure not in ("scale", "nominal", "ordinal"):
        measure = "scale"

    cols_meta[orig_col] = {
        "label": req.label or "",
        "measure": measure,
        "value_labels": req.value_labels or {},
        "missing_codes": req.missing_codes or [],
    }
    meta["columns"] = cols_meta
    session["meta"] = meta
    save_session_to_disk(session)
    sessions[req.session_id] = session

    return {
        "message": f"✅ 已保存列元数据：{cur_col}",
        "meta": _meta_current(session),
        "history": _history_info(session),
        "history_stack": _history_stack(session),
    }


@app.post("/analysis_run")
async def analysis_run_endpoint(req: AnalysisRequest):
    """
    工具面板的“统计/建模”统一入口：确定性计算（表格+图）+ DeepSeek 解释。
    """
    if getattr(req, "runAsync", False):
        task = celery_app.send_task(
            "radarm.analysis",
            kwargs={
                "session_id": req.session_id,
                "analysis": req.analysis,
                "params": req.params or {},
                "explain": bool(req.explain),
                "api_keys": req.apiKeys or {},
                "provider": req.provider or "deepseekA",
                "model": req.model,
            },
        )
        return {"task_id": task.id, "state": "queued", "message": f"分析任务已提交: {req.analysis}"}

    session = get_session_data(req.session_id)
    df = session["df"]

    t0 = time.time()
    try:
        res = run_analysis(session_id=req.session_id, df=df, analysis=req.analysis, params=req.params or {})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"分析执行失败: {str(e)}")

    title = res.get("title", str(req.analysis))
    tables_obj = res.get("tables") or []
    charts_obj = res.get("charts") or []
    summary_obj = res.get("summary") or {}

    tables = []
    for t in tables_obj:
        # dataclass TableOut or dict
        name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else "表格")
        md = getattr(t, "markdown", None) or (t.get("markdown") if isinstance(t, dict) else "")
        tables.append({"name": name, "markdown": md})

    charts = []
    for c in charts_obj:
        name = getattr(c, "name", None) or (c.get("name") if isinstance(c, dict) else "图")
        path = getattr(c, "path", None) or (c.get("path") if isinstance(c, dict) else None)
        if path:
            charts.append({"name": name, "path": path})

    # DeepSeek 解释（可关闭 explain）
    explanation = ""
    if req.explain:
        # 兼容旧字段：primaryModel
        pref_provider = (req.provider or "") if hasattr(req, "provider") else ""
        if not pref_provider:
            pref_provider = (req.primaryModel or "") if hasattr(req, "primaryModel") else ""
        provider = _choose_provider(req.apiKeys or {}, pref_provider or "deepseekA") or "deepseekA"
        api_key = (req.apiKeys or {}).get(provider, "")
        meta_cur = _meta_current(session)
        meta_hint = {}
        try:
            cols_meta = (meta_cur.get("columns") or {})
            for k, v in list(cols_meta.items())[:30]:
                meta_hint[k] = {kk: vv for kk, vv in (v or {}).items() if kk in ("label", "measure")}
        except Exception:
            meta_hint = {}

        def _truncate(s: str, n: int = 3500) -> str:
            s2 = str(s or "")
            return s2 if len(s2) <= n else s2[:n] + "\n...(截断)"

        tables_md = "\n\n".join([f"### {t['name']}\n{t['markdown']}" for t in tables[:3]])
        prompt = (
            "你是一名严谨的统计分析助手。请基于下面给定的分析结果，输出一段中文解释（Markdown）。\n"
            "要求：\n"
            "- 不要杜撰任何数字；所有数字必须来自给定结果。\n"
            "- 先给一句话结论，然后列出关键证据（统计量/df/p 值/样本量/效应量如有）。\n"
            "- 若不显著或数据不足，明确说明，并给出下一步建议。\n"
            "- 语言清晰、可审计。\n\n"
            f"[分析类型]\n{title}\n\n"
            f"[参数(JSON)]\n{_truncate(json.dumps(req.params or {}, ensure_ascii=False, indent=2), 1200)}\n\n"
            f"[数据规模]\n{len(df)} 行, {len(df.columns)} 列\n\n"
            f"[变量元数据(截断)]\n{_truncate(json.dumps(meta_hint, ensure_ascii=False, indent=2), 1200)}\n\n"
            f"[结果摘要(JSON)]\n{_truncate(json.dumps(summary_obj, ensure_ascii=False, indent=2), 2000)}\n\n"
            f"[结果表格(截断)]\n{_truncate(tables_md, 4000)}\n"
        )
        explanation = _call_llm(provider, api_key, prompt, model=req.model, temperature=0.2, timeout=180) or ""

    elapsed_ms = int((time.time() - t0) * 1000)
    return {
        "title": title,
        "reply": explanation or f"✅ 已完成：{title}（未启用解释或模型未返回）",
        "tables": tables,
        "charts": charts,
        "image": charts[0]["path"] if charts else None,
        "data_changed": False,
        "history": _history_info(session),
        "history_stack": _history_stack(session),
        "meta": _meta_current(session),
        "profile": session.get("profile", {}),
        "process_log": json.dumps({"type": "analysis_run", "analysis": req.analysis, "elapsed_ms": elapsed_ms}, ensure_ascii=False),
        "elapsed_ms": elapsed_ms,
    }


@app.post("/onboard_suggest")
async def onboard_suggest_endpoint(req: OnboardSuggestRequest):
    """
    导入数据后的自动“预检 + 建议”：
    - 先跑一次确定性数据概览（表格+缺失率图）
    - 生成一组清洗/变换 Action 建议（需确认）
    - 给出分析建议（描述/差异/相关回归/建模），更接近 SPSS 的工作流
    """
    session = get_session_data(req.session_id)
    df = session.get("df")
    if df is None or not isinstance(df, pd.DataFrame) or len(df.columns) == 0:
        return {"reply": "⚠️ 当前任务还没有可用数据，请先导入数据。", "error": True, "process_log": "onboard_no_df"}

    t0 = time.time()

    # 1) 确定性概览（表+图）
    overview_tables: List[Dict[str, Any]] = []
    overview_charts: List[Dict[str, Any]] = []
    overview_summary: Dict[str, Any] = {}
    try:
        res = run_analysis(session_id=req.session_id, df=df, analysis="overview", params={})
        tables_obj = res.get("tables") or []
        charts_obj = res.get("charts") or []
        overview_summary = res.get("summary") or {}
        for t in tables_obj:
            name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else "表格")
            md = getattr(t, "markdown", None) or (t.get("markdown") if isinstance(t, dict) else "")
            overview_tables.append({"name": name, "markdown": md})
        for c in charts_obj:
            name = getattr(c, "name", None) or (c.get("name") if isinstance(c, dict) else "图")
            path = getattr(c, "path", None) or (c.get("path") if isinstance(c, dict) else None)
            if path:
                overview_charts.append({"name": name, "path": path})
    except Exception:
        overview_tables = []
        overview_charts = []
        overview_summary = {}

    # 2) 清洗/变换建议（Action Pipeline）
    api_keys = req.apiKeys or {}
    prefer_provider = str(req.provider or "deepseekA")
    # 若用户指定 model，仅对该 provider 生效；如果最终兜底到其他 provider，则使用该 provider 的默认模型
    overrides: Dict[str, str] = {}
    if req.model and prefer_provider:
        overrides[prefer_provider] = str(req.model)
    mc = _apply_model_overrides(MODEL_CONFIG, overrides) if overrides else MODEL_CONFIG

    available_cols = [str(c) for c in getattr(df, "columns", [])]
    data_ctx = session.get("profile_context", "") or ""
    clean_query = (
        "这是一次【导入后预检】。\n"
        "请基于数据概况，给出最值得做的 3-6 个清洗/变换 Action 建议（优先：缺失值处理、类型修正、去重、字符串去空格、异常值缩尾、分类变量 one-hot）。\n"
        "输出的 Action 需要尽量具体到列；如果缺少关键信息（如缺失码），请在风险提示里指出需要用户确认。\n"
    )
    clean_sug = suggest_clean_actions(
        user_query=clean_query,
        data_context=data_ctx,
        api_keys=api_keys,
        primary_model=prefer_provider,
        model_config=mc,
        available_columns=available_cols,
    )
    suggested_actions = clean_sug.get("suggested_actions") or []
    risk_notes = clean_sug.get("risk_notes", []) or []

    # 3) 分析建议（AI：基于概览+候选列，给出可执行的下一步）
    health = _dataset_health_summary(df)
    provider_used = str(clean_sug.get("provider") or _choose_provider(api_keys, prefer_provider) or prefer_provider)
    api_key_used = api_keys.get(provider_used, "")
    model_for_reply = str(req.model) if (req.model and provider_used == prefer_provider) else None

    reply_md = ""
    if api_key_used:
        def _truncate(s: str, n: int = 4500) -> str:
            s2 = str(s or "")
            return s2 if len(s2) <= n else s2[:n] + "\n...(截断)"

        tables_md = "\n\n".join([f"### {t['name']}\n{t['markdown']}" for t in (overview_tables or [])[:1]])
        prompt = (
            "你是 Radarm 的【导入后数据预检】助手（更接近 SPSS Pro 的工作流）。\n"
            "用户刚导入一个数据集，你需要先“理解数据”，再给出“清洗/变换建议 + 分析建议”。\n\n"
            "输出要求（Markdown）：\n"
            "1) **数据理解**：一句话总结 + 关键事实（行/列、缺失率 Top、重复行比例、可能的ID列/日期列/二分类列）。\n"
            "2) **清洗/变换建议（需确认）**：用自然语言解释下面给定的 Action 建议卡片（不要新增不同的清洗动作），并补充风险提示。\n"
            "3) **分析建议（下一步）**：按 SPSS 风格给 6-10 条建议，分组为：\n"
            "   - 描述性（概览/频数/描述统计/正态性）\n"
            "   - 差异性（T检验/ANOVA/卡方/非参）\n"
            "   - 相关/回归/建模（相关/线性/逻辑/PCA/KMeans）\n"
            "   每条建议必须给：为什么 + 一条可直接复制到聊天的指令示例（用 @列名）。\n"
            "4) **需要你确认的问题**：列出 2-4 个问题（例如：目标变量/分组列/缺失码/异常值阈值）。\n\n"
            "注意：\n"
            "- 不要杜撰任何具体数值；数值只能来自给定的概览/摘要。\n"
            "- 不要输出思考过程。\n"
            "- 只建议系统已支持的分析：overview/frequency/crosstab/descriptive/group_summary/normality/correlation/ttest/anova/chi_square/nonparam/linear_regression/logistic_regression/pca/kmeans。\n\n"
            f"[数据概况(profile_context)]\n{_truncate(data_ctx, 6500)}\n\n"
            f"[数据预检摘要(JSON)]\n{_truncate(json.dumps(health, ensure_ascii=False, indent=2), 2500)}\n\n"
            f"[概览表(截断)]\n{_truncate(tables_md, 4500)}\n\n"
            f"[清洗Action建议(JSON)]\n{_truncate(json.dumps(suggested_actions, ensure_ascii=False, indent=2), 2500)}\n\n"
            f"[清洗风险提示]\n{_truncate(json.dumps(risk_notes, ensure_ascii=False, indent=2), 1200)}\n"
        )
        reply_md = _call_llm(
            provider_used,
            api_key_used,
            prompt,
            model=model_for_reply,
            temperature=0.2,
            timeout=180,
        ) or ""

    # 没有 key：兜底为规则提示 + 清洗引擎返回的 reply
    if not reply_md.strip():
        shape = health.get("shape") or {}
        reply_md = (
            "## 导入后数据预检（未配置可用 API Key，以下为规则/统计建议）\n\n"
            f"- **规模**：{shape.get('rows','?')} 行，{shape.get('cols','?')} 列\n"
            f"- **重复行比例**：{health.get('dup_pct')}%\n"
            f"- **数值列候选**：{', '.join(health.get('numeric_cols') or []) or '（无）'}\n"
            f"- **分类列候选**：{', '.join(health.get('categorical_cols') or []) or '（无）'}\n\n"
            "### 清洗/变换建议（需确认）\n"
            + (clean_sug.get("reply") or "我给出了一组清洗/变换建议，请确认后应用。")
            + "\n\n### 分析建议（下一步）\n"
            "- 先跑一次：数据概览（工具面板→分析→数据概览，或在聊天输入：`数据概览`）\n"
            "- 对主要分类列做：频数/列联（示例：`对@某列做频数分析`；`@列A @列B 做列联/卡方`）\n"
            "- 对主要数值列做：描述统计/正态性/相关（示例：`对@某列做描述统计`；`@列A @列B 做Pearson相关`）\n"
            "- 若有分组列/结局列，可进一步做：T检验/ANOVA/逻辑回归（示例：`@y @group 做独立样本T检验`；`@y @group 做方差分析`；`@y @x1 @x2 做逻辑回归`）\n"
        )

    elapsed_ms = int((time.time() - t0) * 1000)
    return {
        "reply": reply_md,
        "needs_confirmation": True if suggested_actions else False,
        "suggested_actions": suggested_actions,
        "risk_notes": risk_notes,
        "tables": overview_tables,
        "charts": overview_charts,
        "image": overview_charts[0]["path"] if overview_charts else None,
        "data_changed": False,
        "history": _history_info(session),
        "history_stack": _history_stack(session),
        "meta": _meta_current(session),
        "profile": session.get("profile", {}),
        "process_log": json.dumps({"type": "onboard_suggest", "provider": provider_used, "elapsed_ms": elapsed_ms}, ensure_ascii=False),
        "elapsed_ms": elapsed_ms,
    }


@app.post("/upload_chat_image")
async def upload_chat_image(session_id: str = Form(...), file: UploadFile = File(...)):
    """
    Ask 模式图片上传：保存到 out/{session_id}/，返回相对路径（用于 /out/... 访问）。
    """
    if not file:
        raise HTTPException(status_code=400, detail="缺少图片文件")
    ctype = (file.content_type or "").lower()
    if ctype and not ctype.startswith("image/"):
        raise HTTPException(status_code=400, detail="仅支持图片文件上传")

    out_dir = Path("out") / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "").suffix
    if not ext or len(ext) > 10:
        ext = ".png"
    ts = int(time.time() * 1000)
    fname = f"upload_{ts}{ext}"
    fp = out_dir / fname
    content = await file.read()
    fp.write_bytes(content)

    return {"path": f"{session_id}/{fname}"}

# 静态文件服务：提供 out/ 下任意子路径访问（支持 reports/... 等）
@app.get("/out/{file_path:path}")
async def serve_out_file(file_path: str):
    """
    允许访问 out/{...} 下的文件，用于：
    - 图表 PNG
    - 报告产物（report.md / manifest.json / CSV 等）
    注意：做路径穿越防护，只允许 out 目录内文件。
    """
    root = Path("out").resolve()
    rel = str(file_path or "").lstrip("/\\")
    full = (root / rel).resolve()
    # 防路径穿越：full 必须在 root 下
    if not (str(full) == str(root) or str(full).startswith(str(root) + os.sep)):
        raise HTTPException(status_code=400, detail="非法路径")
    if not full.exists() or not full.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(str(full))

# 删除任务的 out 目录（供前端调用）
@app.post("/cleanup_out")
async def cleanup_out(session_id: str = Form(...)):
    out_dir = Path("out") / session_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    return {"message": f"已清理 {session_id} 的图表目录"}

if __name__ == "__main__":
    import uvicorn
    import signal
    import sys
    
    # 确保 out 目录存在
    Path("out").mkdir(exist_ok=True)
    
    # 优雅关闭处理（环境变量 FOR_DISABLE_CONSOLE_CTRL_HANDLER 已在文件开头设置）
    def signal_handler(sig, frame):
        """处理 CTRL+C 信号，优雅关闭服务器"""
        print("\n\n正在优雅关闭服务器...")
        # 清理 matplotlib 资源
        try:
            plt.close('all')
        except Exception:
            pass
        sys.exit(0)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动服务器
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        # 捕获 KeyboardInterrupt 并优雅退出
        print("\n\n正在优雅关闭服务器...")
        try:
            plt.close('all')
        except Exception:
            pass
        sys.exit(0)
