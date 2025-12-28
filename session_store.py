from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

SESSION_VERSION = 1

_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _now_ts() -> float:
    return float(time.time())


def _safe_session_id(session_id: str) -> str:
    """
    将 session_id 约束为安全的文件夹名，避免路径穿越。
    """
    s = (session_id or "").strip()
    s = _SAFE_ID_RE.sub("_", s)
    s = s[:128] or "default"
    return s


def get_sessions_root() -> Path:
    return Path(__file__).resolve().parent / "radarm_data" / "sessions"


def get_session_dir(session_id: str) -> Path:
    sid = _safe_session_id(session_id)
    return get_sessions_root() / sid


def _meta_path(session_id: str) -> Path:
    return get_session_dir(session_id) / "session.json"


def _raw_path(session_id: str) -> Path:
    return get_session_dir(session_id) / "raw.pkl"


def ensure_storage_ready() -> None:
    get_sessions_root().mkdir(parents=True, exist_ok=True)


def load_session_from_disk(session_id: str) -> Optional[Dict[str, Any]]:
    """
    从磁盘加载 session（只加载 raw_df + 元数据/actions/cursor 等，不计算 current df）。
    若不存在则返回 None。
    """
    ensure_storage_ready()
    meta_p = _meta_path(session_id)
    raw_p = _raw_path(session_id)
    if not meta_p.exists() or not raw_p.exists():
        return None

    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        raw_df = pd.read_pickle(raw_p)
    except Exception:
        return None

    return {
        "version": int(meta.get("version", SESSION_VERSION)),
        "session_id": meta.get("session_id", session_id),
        "filename": meta.get("filename", "data.csv"),
        "created_at": float(meta.get("created_at", _now_ts())),
        "updated_at": float(meta.get("updated_at", _now_ts())),
        "actions": meta.get("actions", []) or [],
        "cursor": int(meta.get("cursor", 0)),
        "meta": meta.get("meta", {}) or {},
        "reports": meta.get("reports", []) or [],
        "agent_pending": meta.get("agent_pending", None),
        "raw_df": raw_df,
    }


def save_session_to_disk(session: Dict[str, Any]) -> None:
    """
    将 session 持久化到磁盘：
    - raw.pkl: raw_df
    - session.json: actions/cursor/meta/filename 等
    """
    ensure_storage_ready()
    session_id = session.get("session_id") or "default"
    sid = _safe_session_id(str(session_id))

    sdir = get_sessions_root() / sid
    sdir.mkdir(parents=True, exist_ok=True)

    # 1) raw_df
    raw_df = session.get("raw_df")
    if raw_df is None:
        raw_df = pd.DataFrame({"info": ["请先上传数据"]})
    raw_tmp = sdir / "raw.pkl.tmp"
    raw_p = sdir / "raw.pkl"
    try:
        raw_df.to_pickle(raw_tmp)
        raw_tmp.replace(raw_p)
    except Exception:
        # 尽量不让保存失败影响主流程
        try:
            if raw_tmp.exists():
                raw_tmp.unlink()
        except Exception:
            pass

    # 2) meta json
    now = _now_ts()
    meta_obj = {
        "version": int(session.get("version", SESSION_VERSION)),
        "session_id": str(session_id),
        "filename": session.get("filename", "data.csv"),
        "created_at": float(session.get("created_at", now)),
        "updated_at": now,
        "actions": session.get("actions", []) or [],
        "cursor": int(session.get("cursor", 0)),
        "meta": session.get("meta", {}) or {},
        "reports": session.get("reports", []) or [],
        "agent_pending": session.get("agent_pending", None),
    }

    meta_tmp = sdir / "session.json.tmp"
    meta_p = sdir / "session.json"
    try:
        meta_tmp.write_text(json.dumps(meta_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        meta_tmp.replace(meta_p)
    except Exception:
        try:
            if meta_tmp.exists():
                meta_tmp.unlink()
        except Exception:
            pass


