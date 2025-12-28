from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional


_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_\-]+")


def safe_id(s: str) -> str:
    t = _SAFE_ID_RE.sub("_", str(s or "").strip())
    t = t.replace("..", "_")[:128]
    return t or "default"


def list_session_files(session_id: str, *, exts: Optional[List[str]] = None, limit: int = 30) -> List[str]:
    """
    列出 out/{session_id}/ 下的文件（递归），返回相对于 out/ 的路径（用于前端 /out/<path>）。
    默认只取 png。
    """
    sid = safe_id(session_id)
    root = Path("out") / sid
    if not root.exists() or not root.is_dir():
        return []

    exts = exts or [".png"]
    exts_l = {e.lower() for e in exts}
    files: List[Path] = []
    try:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() in exts_l:
                files.append(p)
    except Exception:
        return []

    # 最新优先
    try:
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    except Exception:
        pass

    out_root = Path("out")
    res: List[str] = []
    for p in files[: max(1, int(limit))]:
        try:
            res.append(p.relative_to(out_root).as_posix())
        except Exception:
            # fallback
            res.append(str(p).replace("\\", "/"))
    return res


def pick_files_by_query(paths: List[str], query: str, *, limit: int = 6) -> List[str]:
    """
    简单字符串匹配：根据 query 选取文件（用于 reuse_images 工具）。
    """
    q = (query or "").strip().lower()
    if not paths:
        return []
    if not q:
        return paths[: max(1, int(limit))]

    tokens = [t for t in re.split(r"[\s,，;/]+", q) if t]
    scored = []
    for p in paths:
        s = str(p).lower()
        score = 0
        for t in tokens:
            if t in s:
                score += 1
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = [p for _sc, p in scored]
    return out[: max(1, int(limit))]


def build_resource_context(image_paths: List[str], *, max_items: int = 12) -> str:
    if not image_paths:
        return "[已有图表资源]\n- （无）"
    lines = [f"- {p}" for p in image_paths[: max_items]]
    if len(image_paths) > max_items:
        lines.append(f"...（其余 {len(image_paths) - max_items} 个省略）")
    return "[已有图表资源]\n" + "\n".join(lines)


