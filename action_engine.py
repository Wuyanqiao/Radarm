from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


SUPPORTED_ACTION_TYPES = {
    "replace_missing",
    "dropna_rows",
    "fillna",
    "cast_type",
    "standardize",
    "winsorize",
    "one_hot_encode",
    "rename_columns",
    "drop_columns",
    "deduplicate",
    "trim_whitespace",
}


def sanitize_jsonable(value: Any) -> Any:
    """
    将 pandas/numpy 的特殊值转换为 JSON 友好的表示。
    """
    if value is None:
        return None
    if isinstance(value, (np.floating, float)) and (math.isinf(value) or math.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def build_data_profile(df: pd.DataFrame, *, max_cols: int = 50) -> Dict[str, Any]:
    rows = int(len(df))
    cols = int(len(df.columns))

    # 缺失率 Top
    missing_top: List[Dict[str, Any]] = []
    try:
        miss_pct = (df.isna().mean() * 100).sort_values(ascending=False)
        for name, pct in miss_pct.head(15).items():
            missing_top.append({"name": str(name), "missing_pct": float(pct)})
    except Exception:
        pass

    # 列画像（截断）
    columns: List[Dict[str, Any]] = []
    for col in list(df.columns)[:max_cols]:
        try:
            s = df[col]
            dtype = str(s.dtype)
            miss = float(s.isna().mean() * 100) if rows > 0 else 0.0
            nunique = int(s.nunique(dropna=True))
            examples = s.dropna().astype(str).unique()[:3].tolist()
            examples = [e[:30] for e in examples]
            columns.append(
                {
                    "name": str(col),
                    "dtype": dtype,
                    "missing_pct": miss,
                    "nunique": nunique,
                    "examples": examples,
                }
            )
        except Exception:
            columns.append({"name": str(col), "dtype": "unknown", "missing_pct": None, "nunique": None, "examples": []})

    # 样例行
    sample_rows: List[Dict[str, Any]] = []
    try:
        d = df.head(5).replace([float("inf"), float("-inf")], float("nan")).fillna("")
        sample_rows = d.to_dict(orient="records")
    except Exception:
        sample_rows = []

    # 数值摘要（截断列数，避免体积过大）
    numeric_summary: Dict[str, Any] = {}
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols[:10]:
            s = pd.to_numeric(df[col], errors="coerce")
            numeric_summary[str(col)] = {
                "mean": sanitize_jsonable(s.mean()),
                "std": sanitize_jsonable(s.std()),
                "min": sanitize_jsonable(s.min()),
                "max": sanitize_jsonable(s.max()),
            }
    except Exception:
        numeric_summary = {}

    return {
        "shape": {"rows": rows, "cols": cols},
        "missing_top": missing_top,
        "columns": columns,
        "numeric_summary": numeric_summary,
        "sample_rows": sample_rows,
    }


def profile_to_context(profile: Dict[str, Any]) -> str:
    """
    将 profile 转成给 LLM 的紧凑上下文文本。
    """
    shape = profile.get("shape") or {}
    cols = profile.get("columns") or []
    missing_top = profile.get("missing_top") or []
    numeric_summary = profile.get("numeric_summary") or {}
    sample_rows = profile.get("sample_rows") or []

    lines: List[str] = []
    lines.append(f"规模: {shape.get('rows', '?')} 行, {shape.get('cols', '?')} 列")
    if missing_top:
        miss_str = ", ".join([f"{x.get('name')}={x.get('missing_pct'):.1f}%" for x in missing_top if x.get("missing_pct") is not None][:10])
        lines.append(f"缺失率Top: {miss_str}")

    lines.append("字段画像(截断):")
    for c in cols:
        lines.append(
            f"- {c.get('name')} ({c.get('dtype')}), missing={c.get('missing_pct'):.1f}% , unique≈{c.get('nunique')}, examples={c.get('examples')}"
        )

    if numeric_summary:
        lines.append("数值摘要(截断):")
        for k, v in list(numeric_summary.items())[:10]:
            lines.append(f"- {k}: mean={v.get('mean')} std={v.get('std')} min={v.get('min')} max={v.get('max')}")

    if sample_rows:
        lines.append(f"样例行(前5行): {sample_rows}")

    return "\n".join(lines)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "").strip().lower())


def resolve_columns(requested: Iterable[str], available: Iterable[str]) -> List[str]:
    """
    将 LLM 给的列名尽量映射到真实列名（大小写/空格/包含匹配）。
    """
    avail = list(available)
    avail_norm = {_norm(c): c for c in avail}

    resolved: List[str] = []
    for r in requested or []:
        rn = _norm(r)
        if not rn:
            continue
        if rn in avail_norm:
            resolved.append(avail_norm[rn])
            continue
        # 包含匹配（弱匹配）
        hit = None
        for c in avail:
            if rn and rn in _norm(c):
                hit = c
                break
        if hit:
            resolved.append(hit)
    # 去重但保序
    seen = set()
    out = []
    for c in resolved:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def validate_actions(actions: List[Dict[str, Any]]) -> Tuple[bool, str]:
    if not isinstance(actions, list):
        return False, "actions 必须是数组"
    if len(actions) > 20:
        return False, "actions 过多（上限 20）"
    for i, a in enumerate(actions):
        if not isinstance(a, dict):
            return False, f"actions[{i}] 不是对象"
        t = a.get("type")
        if t not in SUPPORTED_ACTION_TYPES:
            return False, f"actions[{i}].type 不支持: {t}"
        if "params" in a and not isinstance(a.get("params"), dict):
            return False, f"actions[{i}].params 必须是对象"
    return True, ""


def _coerce_values(values: List[Any]) -> List[Any]:
    out: List[Any] = []
    for v in values or []:
        if isinstance(v, str):
            s = v.strip()
            # 尝试把纯数字字符串转成数字（999 -> 999）
            if re.fullmatch(r"-?\d+", s):
                try:
                    out.append(int(s))
                    continue
                except Exception:
                    pass
            if re.fullmatch(r"-?\d+\.\d+", s):
                try:
                    out.append(float(s))
                    continue
                except Exception:
                    pass
            out.append(s)
        else:
            out.append(v)
    return out


def apply_action(df: pd.DataFrame, action: Dict[str, Any]) -> pd.DataFrame:
    t = action.get("type")
    params = action.get("params") or {}

    if t == "replace_missing":
        cols = params.get("columns") or []
        values = _coerce_values(params.get("values") or [])
        if not values:
            return df
        if not cols:
            cols = list(df.columns)
        cols = resolve_columns(cols, df.columns)
        out = df.copy()
        for c in cols:
            out[c] = out[c].replace(values, np.nan)
        return out

    if t == "dropna_rows":
        cols = params.get("columns") or []
        how = (params.get("how") or "any").lower()
        if how not in ("any", "all"):
            how = "any"
        subset = resolve_columns(cols, df.columns) if cols else None
        return df.dropna(subset=subset, how=how).reset_index(drop=True)

    if t == "fillna":
        cols = params.get("columns") or []
        strategy = (params.get("strategy") or "value").lower()
        value = params.get("value")
        out = df.copy()
        target_cols = resolve_columns(cols, df.columns) if cols else list(df.columns)
        for c in target_cols:
            s = out[c]
            if strategy == "value":
                out[c] = s.fillna(value)
            elif strategy in ("zero", "0"):
                out[c] = s.fillna(0)
            elif strategy == "mean":
                v = pd.to_numeric(s, errors="coerce").mean()
                out[c] = s.fillna(v)
            elif strategy == "median":
                v = pd.to_numeric(s, errors="coerce").median()
                out[c] = s.fillna(v)
            elif strategy == "mode":
                try:
                    m = s.mode(dropna=True)
                    v = m.iloc[0] if len(m) else ""
                except Exception:
                    v = ""
                out[c] = s.fillna(v)
        return out

    if t == "cast_type":
        col = params.get("column")
        to = (params.get("to") or "").lower()
        if not col:
            return df
        resolved = resolve_columns([col], df.columns)
        if not resolved:
            return df
        c = resolved[0]
        out = df.copy()
        if to in ("float", "double", "number", "numeric"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
        elif to in ("int", "int64", "integer"):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
        elif to in ("str", "string", "text"):
            out[c] = out[c].astype(str)
        elif to in ("category", "cat"):
            out[c] = out[c].astype("category")
        elif to in ("datetime", "date", "time"):
            out[c] = pd.to_datetime(out[c], errors="coerce")
        return out

    if t == "standardize":
        cols = params.get("columns") or []
        method = (params.get("method") or "zscore").lower()
        if method != "zscore":
            method = "zscore"
        out = df.copy()
        target_cols = resolve_columns(cols, df.columns) if cols else []
        for c in target_cols:
            s = pd.to_numeric(out[c], errors="coerce")
            mean = s.mean()
            std = s.std(ddof=0)
            if std == 0 or pd.isna(std):
                out[c] = s - mean
            else:
                out[c] = (s - mean) / std
        return out

    if t == "winsorize":
        cols = params.get("columns") or []
        lower = params.get("lower", 0.01)
        upper = params.get("upper", 0.99)
        try:
            lower = float(lower)
            upper = float(upper)
        except Exception:
            lower, upper = 0.01, 0.99
        lower = max(0.0, min(lower, 0.49))
        upper = max(0.51, min(upper, 1.0))
        if lower >= upper:
            lower, upper = 0.01, 0.99

        out = df.copy()
        target_cols = resolve_columns(cols, df.columns) if cols else []
        for c in target_cols:
            s = pd.to_numeric(out[c], errors="coerce")
            lo = s.quantile(lower)
            hi = s.quantile(upper)
            out[c] = s.clip(lower=lo, upper=hi)
        return out

    if t == "one_hot_encode":
        cols = params.get("columns") or []
        drop_first = bool(params.get("drop_first", False))
        prefix_sep = params.get("prefix_sep", "=")
        if not cols:
            return df
        target_cols = resolve_columns(cols, df.columns)
        if not target_cols:
            return df
        try:
            return pd.get_dummies(df, columns=target_cols, drop_first=drop_first, prefix_sep=prefix_sep)
        except Exception:
            # 兜底：逐列 get_dummies
            out = df.copy()
            for c in target_cols:
                d = pd.get_dummies(out[c].astype(str), prefix=c, prefix_sep=prefix_sep, drop_first=drop_first)
                out = out.drop(columns=[c]).join(d)
            return out

    if t == "rename_columns":
        mapping = params.get("mapping") or {}
        if not isinstance(mapping, dict) or not mapping:
            return df
        # 先做列名解析
        resolved_map: Dict[str, str] = {}
        for k, v in mapping.items():
            if not v:
                continue
            rk = resolve_columns([k], df.columns)
            if rk:
                resolved_map[rk[0]] = str(v)
        if not resolved_map:
            return df
        return df.rename(columns=resolved_map)

    if t == "drop_columns":
        cols = params.get("columns") or []
        target_cols = resolve_columns(cols, df.columns)
        if not target_cols:
            return df
        return df.drop(columns=target_cols)

    if t == "deduplicate":
        subset = params.get("subset") or []
        keep = (params.get("keep") or "first").lower()
        if keep not in ("first", "last"):
            keep = "first"
        subset_cols = resolve_columns(subset, df.columns) if subset else None
        return df.drop_duplicates(subset=subset_cols, keep=keep).reset_index(drop=True)

    if t == "trim_whitespace":
        cols = params.get("columns") or []
        out = df.copy()
        target_cols = resolve_columns(cols, df.columns) if cols else [
            c for c in df.columns if str(df[c].dtype) == "object"
        ]
        for c in target_cols:
            try:
                out[c] = out[c].astype(str).str.strip()
            except Exception:
                continue
        return out

    return df


def apply_actions(raw_df: pd.DataFrame, actions: List[Dict[str, Any]]) -> pd.DataFrame:
    df = raw_df.copy()
    for a in actions or []:
        df = apply_action(df, a)
    df = df.replace([float("inf"), float("-inf")], float("nan"))
    return df


def action_to_brief(action: Dict[str, Any]) -> str:
    t = action.get("type")
    p = action.get("params") or {}
    if t == "replace_missing":
        return f"将 {p.get('columns') or '所有列'} 中的 {p.get('values')} 设为缺失值"
    if t == "dropna_rows":
        return f"删除缺失值行（how={p.get('how','any')}, columns={p.get('columns') or '全部'}）"
    if t == "fillna":
        return f"填充缺失值（strategy={p.get('strategy','value')}, columns={p.get('columns') or '全部'}）"
    if t == "cast_type":
        return f"类型转换：{p.get('column')} -> {p.get('to')}"
    if t == "standardize":
        return f"标准化：{p.get('columns')}"
    if t == "winsorize":
        return f"缩尾处理：{p.get('columns')}（{p.get('lower',0.01)}~{p.get('upper',0.99)}分位）"
    if t == "one_hot_encode":
        return f"虚拟变量转换(one-hot)：{p.get('columns')}（drop_first={p.get('drop_first', False)}）"
    if t == "rename_columns":
        return f"重命名列：{p.get('mapping')}"
    if t == "drop_columns":
        return f"删除列：{p.get('columns')}"
    if t == "deduplicate":
        return f"去重（subset={p.get('subset') or '全部'}, keep={p.get('keep','first')}）"
    if t == "trim_whitespace":
        return f"去除空格：{p.get('columns') or '字符串列'}"
    return str(action)


def actions_to_pandas_code(actions: List[Dict[str, Any]]) -> str:
    """
    生成可复制的 pandas 代码（用于代码透明化展示）。
    注意：这里的 df 假设是当前 DataFrame 变量名。
    """
    lines: List[str] = ["import pandas as pd", "import numpy as np", ""]
    for a in actions or []:
        t = a.get("type")
        p = a.get("params") or {}

        if t == "replace_missing":
            cols = p.get("columns") or []
            values = p.get("values") or []
            if not cols:
                lines.append(f"df = df.replace({values!r}, np.nan)")
            elif len(cols) == 1:
                lines.append(f"df[{cols[0]!r}] = df[{cols[0]!r}].replace({values!r}, np.nan)")
            else:
                lines.append(f"for _c in {cols!r}:")
                lines.append(f"    df[_c] = df[_c].replace({values!r}, np.nan)")

        elif t == "dropna_rows":
            cols = p.get("columns") or []
            how = (p.get("how") or "any").lower()
            if cols:
                lines.append(f"df = df.dropna(subset={cols!r}, how={how!r}).reset_index(drop=True)")
            else:
                lines.append(f"df = df.dropna(how={how!r}).reset_index(drop=True)")

        elif t == "fillna":
            cols = p.get("columns") or []
            strategy = (p.get("strategy") or "value").lower()
            value = p.get("value")
            if not cols:
                lines.append("# 注意：此处未指定 columns，建议按列逐个填充以避免误伤")
            if strategy == "value":
                for c in cols:
                    lines.append(f"df[{c!r}] = df[{c!r}].fillna({value!r})")
            elif strategy in ("zero", "0"):
                for c in cols:
                    lines.append(f"df[{c!r}] = df[{c!r}].fillna(0)")
            elif strategy in ("mean", "median"):
                func = "mean" if strategy == "mean" else "median"
                for c in cols:
                    lines.append(f"df[{c!r}] = df[{c!r}].fillna(pd.to_numeric(df[{c!r}], errors='coerce').{func}())")
            elif strategy == "mode":
                for c in cols:
                    lines.append(f"df[{c!r}] = df[{c!r}].fillna(df[{c!r}].mode(dropna=True).iloc[0] if len(df[{c!r}].mode(dropna=True)) else '')")

        elif t == "cast_type":
            col = p.get("column")
            to = (p.get("to") or "").lower()
            if to in ("float", "double", "number", "numeric"):
                lines.append(f"df[{col!r}] = pd.to_numeric(df[{col!r}], errors='coerce')")
            elif to in ("int", "int64", "integer"):
                lines.append(f"df[{col!r}] = pd.to_numeric(df[{col!r}], errors='coerce').astype('Int64')")
            elif to in ("str", "string", "text"):
                lines.append(f"df[{col!r}] = df[{col!r}].astype(str)")
            elif to in ("category", "cat"):
                lines.append(f"df[{col!r}] = df[{col!r}].astype('category')")
            elif to in ("datetime", "date", "time"):
                lines.append(f"df[{col!r}] = pd.to_datetime(df[{col!r}], errors='coerce')")

        elif t == "standardize":
            cols = p.get("columns") or []
            for c in cols:
                lines.append(f"_s = pd.to_numeric(df[{c!r}], errors='coerce')")
                lines.append(f"_std = _s.std(ddof=0)")
                lines.append(f"df[{c!r}] = (_s - _s.mean()) / _std if _std not in (0, None) else (_s - _s.mean())")

        elif t == "winsorize":
            cols = p.get("columns") or []
            lower = float(p.get("lower", 0.01))
            upper = float(p.get("upper", 0.99))
            for c in cols:
                lines.append(f"_s = pd.to_numeric(df[{c!r}], errors='coerce')")
                lines.append(f"_lo = _s.quantile({lower})")
                lines.append(f"_hi = _s.quantile({upper})")
                lines.append(f"df[{c!r}] = _s.clip(lower=_lo, upper=_hi)")

        elif t == "one_hot_encode":
            cols = p.get("columns") or []
            drop_first = bool(p.get("drop_first", False))
            prefix_sep = p.get("prefix_sep", "=")
            lines.append(f"df = pd.get_dummies(df, columns={cols!r}, drop_first={drop_first!r}, prefix_sep={prefix_sep!r})")

        elif t == "rename_columns":
            mapping = p.get("mapping") or {}
            lines.append(f"df = df.rename(columns={mapping!r})")

        elif t == "drop_columns":
            cols = p.get("columns") or []
            lines.append(f"df = df.drop(columns={cols!r})")

        elif t == "deduplicate":
            subset = p.get("subset") or []
            keep = (p.get("keep") or "first").lower()
            if subset:
                lines.append(f"df = df.drop_duplicates(subset={subset!r}, keep={keep!r}).reset_index(drop=True)")
            else:
                lines.append(f"df = df.drop_duplicates(keep={keep!r}).reset_index(drop=True)")

        elif t == "trim_whitespace":
            cols = p.get("columns") or []
            for c in cols:
                lines.append(f"df[{c!r}] = df[{c!r}].astype(str).str.strip()")

        else:
            lines.append(f"# 未支持的 action: {a!r}")

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


