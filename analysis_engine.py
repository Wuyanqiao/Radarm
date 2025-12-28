from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import re


@dataclass
class TableOut:
    name: str
    markdown: str
    df: Optional[pd.DataFrame] = None
    csv: Optional[str] = None  # relative path under /out (optional)


@dataclass
class ChartOut:
    name: str
    path: str  # relative: "{session_id}/{filename}.png"


def _safe_to_markdown(df: pd.DataFrame, *, max_rows: int = 60) -> str:
    d = df.copy()
    if len(d) > max_rows:
        d = d.head(max_rows)
        note = f"\n\n> 注：表格过长已截断，仅展示前 {max_rows} 行。"
    else:
        note = ""
    try:
        md = d.to_markdown(index=False)
    except Exception:
        md = d.to_string(index=False)
    return md + note


_SAFE_PATH_RE = re.compile(r"[^a-zA-Z0-9_/\-]+")


def _safe_rel_path(p: str) -> str:
    s = str(p or "").strip().replace("\\", "/")
    s = _SAFE_PATH_RE.sub("_", s)
    # 防止路径穿越
    s = s.replace("..", "_")
    s = s.strip("/")
    return s or "default"


def _ensure_out_dir(session_id: str, out_subdir: Optional[str] = None) -> Path:
    sid = _safe_rel_path(session_id)
    sub = _safe_rel_path(out_subdir) if out_subdir else ""
    out_dir = (Path("out") / sid / sub) if sub else (Path("out") / sid)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_current_fig(session_id: str, *, prefix: str = "chart", out_subdir: Optional[str] = None) -> Optional[str]:
    """
    保存当前 matplotlib figure（调用方需先画图并保证存在 figure）。
    返回相对路径："{session_id}/{filename}"
    """
    import matplotlib.pyplot as plt

    if not plt.get_fignums():
        return None
    out_dir = _ensure_out_dir(session_id, out_subdir)
    ts = int(time.time() * 1000)
    filename = f"{prefix}_{ts}.png"
    path = out_dir / filename
    plt.savefig(str(path), format="png", bbox_inches="tight", dpi=200)
    plt.close()
    sid = _safe_rel_path(session_id)
    sub = _safe_rel_path(out_subdir) if out_subdir else ""
    return f"{sid}/{sub}/{filename}" if sub else f"{sid}/{filename}"


def _to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _pick_top2_groups(s: pd.Series) -> Tuple[Optional[Any], Optional[Any]]:
    try:
        vc = s.value_counts(dropna=True)
        vals = vc.index.tolist()
        if len(vals) >= 2:
            return vals[0], vals[1]
        if len(vals) == 1:
            return vals[0], None
    except Exception:
        pass
    return None, None


def analysis_overview(df: pd.DataFrame) -> Tuple[List[TableOut], List[ChartOut], Dict[str, Any]]:
    rows, cols = int(len(df)), int(len(df.columns))
    col_rows: List[Dict[str, Any]] = []
    for c in df.columns:
        s = df[c]
        try:
            miss = float(s.isna().mean() * 100) if rows > 0 else 0.0
        except Exception:
            miss = None
        try:
            nunique = int(s.nunique(dropna=True))
        except Exception:
            nunique = None
        col_rows.append(
            {
                "列名": str(c),
                "dtype": str(s.dtype),
                "缺失%": None if miss is None else round(miss, 2),
                "唯一值": nunique,
            }
        )
    table = pd.DataFrame(col_rows).sort_values(by=["缺失%"], ascending=False)
    tables = [TableOut(name="数据概览", markdown=_safe_to_markdown(table, max_rows=80), df=table)]
    summary = {"shape": {"rows": rows, "cols": cols}, "missing_top": table.head(10).to_dict(orient="records")}
    # chart: 缺失率 Top
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plot_df = table.dropna(subset=["缺失%"]).head(15).copy()
        if len(plot_df) > 0:
            plot_df = plot_df.sort_values(by="缺失%", ascending=True)
            plt.figure(figsize=(8, max(3, min(10, 0.35 * len(plot_df)))))
            sns.barplot(data=plot_df, x="缺失%", y="列名", color="#F59E0B")
            plt.title("缺失率 Top15（%）")
            plt.tight_layout()
    except Exception:
        pass
    return tables, [], summary


def analysis_frequency(df: pd.DataFrame, *, column: str, top_n: int = 30) -> Tuple[List[TableOut], List[ChartOut], Dict[str, Any]]:
    s = df[column].fillna("缺失").astype(str)
    vc = s.value_counts(dropna=False).head(top_n)
    total = int(len(s))
    freq = pd.DataFrame({"值": vc.index.astype(str), "频数": vc.values})
    freq["占比%"] = (freq["频数"] / max(1, total) * 100).round(2)

    # chart
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, max(3, min(10, 0.35 * len(freq)))))
    sns.barplot(data=freq, y="值", x="频数", color="#4F46E5")
    plt.title(f"频数分析：{column}（Top {len(freq)}）")
    plt.tight_layout()
    # 注意：本函数只负责画图，不保存；由 run_analysis 统一保存（因为 session_id 只在上层可用）。
    tables = [TableOut(name=f"频数：{column}", markdown=_safe_to_markdown(freq, max_rows=60), df=freq)]
    summary = {"column": column, "top": freq.head(10).to_dict(orient="records")}
    return tables, [], summary


def analysis_crosstab(df: pd.DataFrame, *, row: str, col: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    a = df[row].fillna("缺失").astype(str)
    b = df[col].fillna("缺失").astype(str)
    ct = pd.crosstab(a, b, dropna=False)
    chi2, p, dof, expected = stats.chi2_contingency(ct.values)
    summary = {"row": row, "col": col, "chi2": float(chi2), "p": float(p), "dof": int(dof)}
    return ct, summary


def analysis_descriptive(df: pd.DataFrame, *, columns: Optional[List[str]] = None) -> Tuple[List[TableOut], List[ChartOut], Dict[str, Any]]:
    if not columns:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        cols = [c for c in columns if c in df.columns]
    if not cols:
        return [TableOut(name="描述统计", markdown="⚠️ 未找到数值列。")], [], {"error": "no_numeric_columns"}

    d = df[cols].copy()
    desc = d.describe().T
    desc["missing_pct"] = (d.isna().mean() * 100).round(2)
    desc["median"] = d.median(numeric_only=True)
    out = desc.reset_index().rename(columns={"index": "列名"})
    tables = [TableOut(name="描述统计（数值列）", markdown=_safe_to_markdown(out, max_rows=80), df=out)]
    summary = {"columns": cols, "shape": {"rows": int(len(df)), "cols": int(len(df.columns))}}
    # chart: 箱线图（前10列，采样以保证性能）
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plot_cols = cols[:10]
        dplot = df[plot_cols].copy()
        if len(dplot) > 5000:
            dplot = dplot.sample(n=5000, random_state=42)
        dplot = dplot.apply(pd.to_numeric, errors="coerce")
        melted = dplot.melt(var_name="变量", value_name="值").dropna()
        if len(melted) > 0:
            plt.figure(figsize=(max(6, 0.8 * len(plot_cols) + 2), 4))
            sns.boxplot(data=melted, x="变量", y="值", color="#A7F3D0")
            plt.xticks(rotation=20)
            plt.title("数值列分布（箱线图，Top10列）")
            plt.tight_layout()
    except Exception:
        pass
    return tables, [], summary


def analysis_group_summary(
    df: pd.DataFrame, *, group_by: str, metric: str, agg: str = "mean"
) -> Tuple[List[TableOut], List[ChartOut], Dict[str, Any]]:
    if group_by not in df.columns or metric not in df.columns:
        return [TableOut(name="分类汇总", markdown="⚠️ group_by 或 metric 列不存在。")], [], {"error": "missing_columns"}
    s = _to_numeric_series(df, metric)
    g = df[group_by].fillna("缺失").astype(str)
    tmp = pd.DataFrame({group_by: g, metric: s})
    if agg not in ("mean", "median", "sum", "count"):
        agg = "mean"
    grouped = tmp.groupby(group_by)[metric].agg(agg).reset_index().sort_values(by=metric, ascending=False)
    tables = [TableOut(name=f"分类汇总：{group_by} → {agg}({metric})", markdown=_safe_to_markdown(grouped, max_rows=80), df=grouped)]
    summary = {"group_by": group_by, "metric": metric, "agg": agg, "top": grouped.head(10).to_dict(orient="records")}
    # chart: barplot（Top20）
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plot_df = grouped.head(20).copy()
        if len(plot_df) > 0:
            plot_df = plot_df.sort_values(by=metric, ascending=True)
            plt.figure(figsize=(8, max(3, min(10, 0.35 * len(plot_df)))))
            sns.barplot(data=plot_df, x=metric, y=group_by, color="#10B981")
            plt.title(f"{agg}({metric}) by {group_by}（Top20）")
            plt.tight_layout()
    except Exception:
        pass
    return tables, [], summary


def analysis_normality(df: pd.DataFrame, *, column: str, method: str = "auto") -> Tuple[List[TableOut], List[ChartOut], Dict[str, Any]]:
    if column not in df.columns:
        return [TableOut(name="正态性检验", markdown="⚠️ 列不存在。")], [], {"error": "missing_column"}
    s = _to_numeric_series(df, column).dropna()
    n = int(len(s))
    if n < 3:
        return [TableOut(name="正态性检验", markdown="⚠️ 有效样本量不足（n<3）。")], [], {"error": "n_too_small"}

    method = (method or "auto").lower()
    if method == "auto":
        method = "shapiro" if n <= 5000 else "normaltest"

    stat_v, p_v = None, None
    if method == "shapiro":
        stat_v, p_v = stats.shapiro(s.sample(n=min(n, 5000), random_state=42) if n > 5000 else s)
    elif method == "normaltest":
        stat_v, p_v = stats.normaltest(s)
    elif method == "jarque_bera":
        stat_v, p_v = stats.jarque_bera(s)
    else:
        stat_v, p_v = stats.shapiro(s.sample(n=min(n, 5000), random_state=42) if n > 5000 else s)
        method = "shapiro"

    res = pd.DataFrame(
        [
            {"列": column, "n": n, "方法": method, "统计量": float(stat_v), "p值": float(p_v), "结论(α=0.05)": "拒绝正态" if p_v < 0.05 else "未拒绝正态"}
        ]
    )

    # chart: histogram + QQ
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(s.values, bins=30, color="#60A5FA", edgecolor="white")
    plt.title(f"直方图：{column}")
    plt.subplot(1, 2, 2)
    stats.probplot(s.values, dist="norm", plot=plt)
    plt.title(f"Q-Q图：{column}")
    plt.tight_layout()

    tables = [TableOut(name=f"正态性检验：{column}", markdown=_safe_to_markdown(res, max_rows=20), df=res)]
    summary = {"column": column, "n": n, "method": method, "stat": float(stat_v), "p": float(p_v)}
    return tables, [], summary


def analysis_ttest(
    df: pd.DataFrame,
    *,
    ttype: str,
    y: str,
    group_col: Optional[str] = None,
    group_a: Optional[str] = None,
    group_b: Optional[str] = None,
    y2: Optional[str] = None,
    mu: float = 0.0,
) -> Tuple[List[TableOut], List[ChartOut], Dict[str, Any]]:
    ttype = (ttype or "").lower()
    if y not in df.columns:
        return [TableOut(name="T检验", markdown="⚠️ 目标列不存在。")], [], {"error": "missing_y"}

    import matplotlib.pyplot as plt
    import seaborn as sns

    if ttype == "one_sample":
        s = _to_numeric_series(df, y).dropna()
        stat_v, p_v = stats.ttest_1samp(s, popmean=float(mu))
        out = pd.DataFrame([{"检验": "单样本T", "列": y, "mu": float(mu), "n": int(len(s)), "t": float(stat_v), "p": float(p_v)}])
        # chart: histogram + mu line
        try:
            plt.figure(figsize=(6, 4))
            plt.hist(s.values, bins=30, color="#60A5FA", edgecolor="white")
            plt.axvline(float(mu), color="red", linestyle="--", linewidth=1, label=f"mu={float(mu)}")
            plt.title(f"分布：{y}（单样本T）")
            plt.legend()
            plt.tight_layout()
        except Exception:
            pass
        tables = [TableOut(name="单样本T检验", markdown=_safe_to_markdown(out, max_rows=20), df=out)]
        summary = out.iloc[0].to_dict()
        return tables, [], summary

    if ttype == "paired":
        if not y2 or y2 not in df.columns:
            return [TableOut(name="配对T检验", markdown="⚠️ 需要第二列 y2。")], [], {"error": "missing_y2"}
        a = _to_numeric_series(df, y)
        b = _to_numeric_series(df, y2)
        mask = ~(a.isna() | b.isna())
        a2, b2 = a[mask], b[mask]
        stat_v, p_v = stats.ttest_rel(a2, b2)
        out = pd.DataFrame([{"检验": "配对样本T", "列A": y, "列B": y2, "n": int(len(a2)), "t": float(stat_v), "p": float(p_v)}])
        # chart: paired scatter
        plt.figure(figsize=(5, 5))
        plt.scatter(a2, b2, alpha=0.6)
        lo = float(min(a2.min(), b2.min()))
        hi = float(max(a2.max(), b2.max()))
        plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        plt.xlabel(y)
        plt.ylabel(y2)
        plt.title("配对散点图")
        plt.tight_layout()
        tables = [TableOut(name="配对样本T检验", markdown=_safe_to_markdown(out, max_rows=20), df=out)]
        summary = out.iloc[0].to_dict()
        return tables, [], summary

    # independent (Welch)
    if not group_col or group_col not in df.columns:
        return [TableOut(name="独立样本T检验", markdown="⚠️ 需要分组列 group_col。")], [], {"error": "missing_group"}
    g = df[group_col].astype(str).fillna("缺失")
    s = _to_numeric_series(df, y)
    if group_a is None or group_b is None:
        a0, b0 = _pick_top2_groups(g)
        group_a = str(a0) if a0 is not None else None
        group_b = str(b0) if b0 is not None else None
    if group_a is None or group_b is None:
        return [TableOut(name="独立样本T检验", markdown="⚠️ 分组不足两个水平。")], [], {"error": "group_levels<2"}

    a = s[g == str(group_a)].dropna()
    b = s[g == str(group_b)].dropna()
    stat_v, p_v = stats.ttest_ind(a, b, equal_var=False)
    out = pd.DataFrame(
        [
            {
                "检验": "独立样本T(Welch)",
                "列": y,
                "分组列": group_col,
                "组A": str(group_a),
                "nA": int(len(a)),
                "meanA": float(a.mean()) if len(a) else np.nan,
                "组B": str(group_b),
                "nB": int(len(b)),
                "meanB": float(b.mean()) if len(b) else np.nan,
                "t": float(stat_v),
                "p": float(p_v),
            }
        ]
    )
    # chart: boxplot
    plot_df = pd.DataFrame({group_col: g, y: s})
    plot_df = plot_df[plot_df[group_col].isin([str(group_a), str(group_b)])]
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=plot_df, x=group_col, y=y, palette="Set2")
    plt.title(f"箱线图：{y} by {group_col}")
    plt.tight_layout()
    tables = [TableOut(name="独立样本T检验", markdown=_safe_to_markdown(out, max_rows=20), df=out)]
    summary = out.iloc[0].to_dict()
    return tables, [], summary


def analysis_anova_oneway(df: pd.DataFrame, *, y: str, group_col: str) -> Tuple[List[TableOut], List[ChartOut], Dict[str, Any]]:
    if y not in df.columns or group_col not in df.columns:
        return [TableOut(name="单因素方差分析", markdown="⚠️ 列不存在。")], [], {"error": "missing_columns"}
    import matplotlib.pyplot as plt
    import seaborn as sns

    g = df[group_col].astype(str).fillna("缺失")
    s = _to_numeric_series(df, y)
    tmp = pd.DataFrame({group_col: g, y: s}).dropna()
    groups = [vals[y].values for _, vals in tmp.groupby(group_col)]
    if len(groups) < 2:
        return [TableOut(name="单因素方差分析", markdown="⚠️ 分组不足两个水平。")], [], {"error": "group_levels<2"}
    f, p = stats.f_oneway(*groups)
    out = pd.DataFrame([{"检验": "单因素ANOVA", "y": y, "group": group_col, "k": int(len(groups)), "F": float(f), "p": float(p)}])

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=tmp, x=group_col, y=y, palette="Set3")
    plt.title(f"箱线图：{y} by {group_col}")
    plt.xticks(rotation=20)
    plt.tight_layout()

    tables = [TableOut(name="单因素方差分析", markdown=_safe_to_markdown(out, max_rows=20), df=out)]
    summary = out.iloc[0].to_dict()
    return tables, [], summary


def analysis_chi_square(df: pd.DataFrame, *, row: str, col: str) -> Tuple[List[TableOut], List[ChartOut], Dict[str, Any]]:
    ct, summ = analysis_crosstab(df, row=row, col=col)
    ct_df = ct.reset_index()
    stat_df = pd.DataFrame([summ])
    tbl_ct = TableOut(name=f"列联表：{row} x {col}", markdown=_safe_to_markdown(ct_df, max_rows=80), df=ct_df)
    tbl_stat = TableOut(name="卡方检验", markdown=_safe_to_markdown(stat_df, max_rows=10), df=stat_df)

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 5))
    sns.heatmap(ct, annot=False, cmap="Blues")
    plt.title(f"列联热力图：{row} x {col}")
    plt.tight_layout()

    return [tbl_ct, tbl_stat], [], summ


def analysis_nonparam(
    df: pd.DataFrame,
    *,
    test: str,
    y: Optional[str] = None,
    group_col: Optional[str] = None,
    group_a: Optional[str] = None,
    group_b: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> Tuple[List[TableOut], List[ChartOut], Dict[str, Any]]:
    test = (test or "").lower()
    import matplotlib.pyplot as plt
    import seaborn as sns

    if test in ("mann_whitney", "mannwhitney"):
        if not y or not group_col:
            return [TableOut(name="Mann-Whitney", markdown="⚠️ 需要 y 与 group_col。")], [], {"error": "missing_params"}
        g = df[group_col].astype(str).fillna("缺失")
        s = _to_numeric_series(df, y)
        if group_a is None or group_b is None:
            a0, b0 = _pick_top2_groups(g)
            group_a = str(a0) if a0 is not None else None
            group_b = str(b0) if b0 is not None else None
        if group_a is None or group_b is None:
            return [TableOut(name="Mann-Whitney", markdown="⚠️ 分组不足两个水平。")], [], {"error": "group_levels<2"}
        a = s[g == str(group_a)].dropna()
        b = s[g == str(group_b)].dropna()
        stat_v, p_v = stats.mannwhitneyu(a, b, alternative="two-sided")
        out = pd.DataFrame([{"检验": "Mann-Whitney U", "y": y, "group": group_col, "A": group_a, "nA": int(len(a)), "B": group_b, "nB": int(len(b)), "U": float(stat_v), "p": float(p_v)}])

        plot_df = pd.DataFrame({group_col: g, y: s})
        plot_df = plot_df[plot_df[group_col].isin([str(group_a), str(group_b)])]
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=plot_df, x=group_col, y=y, palette="Set2")
        plt.title(f"箱线图：{y} by {group_col}")
        plt.tight_layout()

        return [TableOut(name="Mann-Whitney U检验", markdown=_safe_to_markdown(out, max_rows=20), df=out)], [], out.iloc[0].to_dict()

    if test in ("kruskal", "kruskal_wallis"):
        if not y or not group_col:
            return [TableOut(name="Kruskal-Wallis", markdown="⚠️ 需要 y 与 group_col。")], [], {"error": "missing_params"}
        g = df[group_col].astype(str).fillna("缺失")
        s = _to_numeric_series(df, y)
        tmp = pd.DataFrame({group_col: g, y: s}).dropna()
        groups = [vals[y].values for _, vals in tmp.groupby(group_col)]
        if len(groups) < 2:
            return [TableOut(name="Kruskal-Wallis", markdown="⚠️ 分组不足两个水平。")], [], {"error": "group_levels<2"}
        stat_v, p_v = stats.kruskal(*groups)
        out = pd.DataFrame([{"检验": "Kruskal-Wallis", "y": y, "group": group_col, "k": int(len(groups)), "H": float(stat_v), "p": float(p_v)}])

        plt.figure(figsize=(7, 4))
        sns.boxplot(data=tmp, x=group_col, y=y, palette="Set3")
        plt.title(f"箱线图：{y} by {group_col}")
        plt.xticks(rotation=20)
        plt.tight_layout()

        return [TableOut(name="Kruskal-Wallis检验", markdown=_safe_to_markdown(out, max_rows=20), df=out)], [], out.iloc[0].to_dict()

    if test in ("friedman",):
        cols = columns or []
        cols = [c for c in cols if c in df.columns]
        if len(cols) < 3:
            return [TableOut(name="Friedman", markdown="⚠️ 需要至少 3 个列（重复测量）。")], [], {"error": "need>=3_columns"}
        mat = [pd.to_numeric(df[c], errors="coerce") for c in cols]
        tmp = pd.concat(mat, axis=1)
        tmp.columns = cols
        tmp = tmp.dropna()
        arrays = [tmp[c].values for c in cols]
        stat_v, p_v = stats.friedmanchisquare(*arrays)
        out = pd.DataFrame([{"检验": "Friedman", "k": len(cols), "n": int(len(tmp)), "Q": float(stat_v), "p": float(p_v)}])
        # chart: boxplot for repeated measures
        try:
            import seaborn as sns

            melted = tmp.melt(var_name="变量", value_name="值").dropna()
            if len(melted) > 0:
                plt.figure(figsize=(max(6, 0.8 * len(cols) + 2), 4))
                sns.boxplot(data=melted, x="变量", y="值", color="#FDE68A")
                plt.xticks(rotation=20)
                plt.title("重复测量分布（箱线图）")
                plt.tight_layout()
        except Exception:
            pass
        return [TableOut(name="Friedman检验", markdown=_safe_to_markdown(out, max_rows=10), df=out)], [], out.iloc[0].to_dict()

    return [TableOut(name="非参数检验", markdown="⚠️ 未支持的非参数检验类型。")], [], {"error": "unsupported_test"}


def analysis_correlation(
    df: pd.DataFrame, *, method: str = "pearson", columns: Optional[List[str]] = None
) -> Tuple[List[TableOut], List[ChartOut], Dict[str, Any]]:
    method = (method or "pearson").lower()
    if method not in ("pearson", "spearman", "kendall"):
        method = "pearson"

    if not columns:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        cols = [c for c in columns if c in df.columns]
    if len(cols) < 2:
        return [TableOut(name="相关性分析", markdown="⚠️ 需要至少 2 个数值列。")], [], {"error": "need>=2_columns"}

    x = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = x.corr(method=method)

    # p-values matrix
    pmat = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a = x[cols[i]]
            b = x[cols[j]]
            mask = ~(a.isna() | b.isna())
            a2, b2 = a[mask].values, b[mask].values
            if len(a2) < 3:
                p = np.nan
            else:
                if method == "pearson":
                    _, p = stats.pearsonr(a2, b2)
                elif method == "spearman":
                    _, p = stats.spearmanr(a2, b2)
                else:
                    _, p = stats.kendalltau(a2, b2)
            pmat.iat[i, j] = p
            pmat.iat[j, i] = p

    # top pairs
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append({"A": cols[i], "B": cols[j], "corr": float(corr.iat[i, j]), "p": float(pmat.iat[i, j]) if not pd.isna(pmat.iat[i, j]) else np.nan})
    top = pd.DataFrame(pairs).sort_values(by="corr", ascending=False).head(20)

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap="RdBu_r", center=0, linewidths=0.5)
    plt.title(f"{method.title()} 相关矩阵")
    plt.tight_layout()

    corr_tbl = corr.reset_index().rename(columns={"index": "列名"})
    tables = [
        TableOut(name=f"{method.title()} 相关矩阵", markdown=_safe_to_markdown(corr_tbl, max_rows=80), df=corr_tbl),
        TableOut(name="相关 Top20", markdown=_safe_to_markdown(top, max_rows=30), df=top),
    ]
    summary = {"method": method, "columns": cols, "top": top.head(10).to_dict(orient="records")}
    return tables, [], summary


def analysis_linear_regression(df: pd.DataFrame, *, y: str, x: List[str]) -> Tuple[List[TableOut], List[ChartOut], Dict[str, Any]]:
    if y not in df.columns:
        return [TableOut(name="线性回归", markdown="⚠️ y 列不存在。")], [], {"error": "missing_y"}
    x = [c for c in (x or []) if c in df.columns and c != y]
    if not x:
        return [TableOut(name="线性回归", markdown="⚠️ 需要至少 1 个自变量列。")], [], {"error": "missing_x"}

    y_num = pd.to_numeric(df[y], errors="coerce")
    X_df = df[x].apply(pd.to_numeric, errors="coerce")
    mask = ~(y_num.isna() | X_df.isna().any(axis=1))
    y2 = y_num[mask].to_numpy(dtype=float)
    X2 = X_df[mask].to_numpy(dtype=float)
    n = int(len(y2))
    if n < 5:
        return [TableOut(name="线性回归", markdown="⚠️ 有效样本量不足（n<5）。")], [], {"error": "n_too_small"}

    # add intercept
    Xw = np.column_stack([np.ones(n), X2])
    beta = np.linalg.lstsq(Xw, y2, rcond=None)[0]
    y_pred = Xw @ beta
    resid = y2 - y_pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y2 - float(np.mean(y2))) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    k = len(beta) - 1
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else r2

    mse = ss_res / (n - k - 1) if n > k + 1 else ss_res / max(1, n - 1)
    try:
        var_beta = mse * np.linalg.inv(Xw.T @ Xw)
        se = np.sqrt(np.diag(var_beta))
        tstats = beta / se
        pvals = 2 * (1 - stats.t.cdf(np.abs(tstats), n - k - 1))
        tcrit = stats.t.ppf(0.975, n - k - 1) if n > k + 1 else 1.96
        ci_lo = beta - tcrit * se
        ci_hi = beta + tcrit * se
    except Exception:
        se = np.full(len(beta), np.nan)
        pvals = np.full(len(beta), np.nan)
        ci_lo = np.full(len(beta), np.nan)
        ci_hi = np.full(len(beta), np.nan)

    rows = []
    rows.append({"变量": "截距", "系数": beta[0], "p": pvals[0], "95%CI": f"[{ci_lo[0]:.4f},{ci_hi[0]:.4f}]"})
    for i, name in enumerate(x):
        idx = i + 1
        rows.append({"变量": name, "系数": beta[idx], "p": pvals[idx], "95%CI": f"[{ci_lo[idx]:.4f},{ci_hi[idx]:.4f}]"})
    coef_df = pd.DataFrame(rows)
    meta = pd.DataFrame([{"y": y, "x": ", ".join(x), "n": n, "R2": r2, "Adj_R2": adj_r2}])

    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 4))
    plt.scatter(y2, y_pred, alpha=0.6)
    lo = float(min(y2.min(), y_pred.min()))
    hi = float(max(y2.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    plt.xlabel("实际值")
    plt.ylabel("预测值")
    plt.title("预测值 vs 实际值")
    plt.tight_layout()

    tables = [
        TableOut(name="线性回归：模型信息", markdown=_safe_to_markdown(meta, max_rows=10), df=meta),
        TableOut(name="线性回归：系数表", markdown=_safe_to_markdown(coef_df, max_rows=50), df=coef_df),
    ]
    summary = {"y": y, "x": x, "n": n, "r2": float(r2), "adj_r2": float(adj_r2), "coef": coef_df.to_dict(orient="records")}
    return tables, [], summary


def analysis_pca(df: pd.DataFrame, *, columns: List[str], n_components: int = 2) -> Tuple[List[TableOut], List[ChartOut], Dict[str, Any]]:
    cols = [c for c in (columns or []) if c in df.columns]
    if len(cols) < 2:
        return [TableOut(name="PCA", markdown="⚠️ 需要至少 2 个数值列。")], [], {"error": "need>=2_columns"}
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return [TableOut(name="PCA", markdown="⚠️ 缺少 scikit-learn 依赖。")], [], {"error": "missing_sklearn"}

    X = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(X) < 5:
        return [TableOut(name="PCA", markdown="⚠️ 有效样本不足（n<5）。")], [], {"error": "n_too_small"}
    n_components = int(max(2, min(int(n_components or 2), len(cols))))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(Xs)
    evr = pca.explained_variance_ratio_

    ev_tbl = pd.DataFrame({"成分": [f"PC{i+1}" for i in range(n_components)], "解释方差比": evr, "累计解释方差比": np.cumsum(evr)})
    loadings = pd.DataFrame(pca.components_.T, index=cols, columns=[f"PC{i+1}" for i in range(n_components)]).reset_index().rename(columns={"index": "变量"})

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, n_components + 1), evr, marker="o")
    plt.title("Scree Plot（解释方差比）")
    plt.xlabel("主成分")
    plt.ylabel("解释方差比")
    plt.subplot(1, 2, 2)
    plt.scatter(Z[:, 0], Z[:, 1], alpha=0.6)
    plt.title("PCA 投影（PC1 vs PC2）")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()

    tables = [
        TableOut(name="PCA：解释方差", markdown=_safe_to_markdown(ev_tbl, max_rows=20), df=ev_tbl),
        TableOut(name="PCA：载荷（loadings）", markdown=_safe_to_markdown(loadings, max_rows=60), df=loadings),
    ]
    summary = {"columns": cols, "n": int(len(X)), "n_components": n_components, "explained_variance_ratio": evr.tolist()}
    return tables, [], summary


def analysis_kmeans(df: pd.DataFrame, *, columns: List[str], k: int = 3) -> Tuple[List[TableOut], List[ChartOut], Dict[str, Any]]:
    cols = [c for c in (columns or []) if c in df.columns]
    if len(cols) < 2:
        return [TableOut(name="KMeans", markdown="⚠️ 需要至少 2 个数值列。")], [], {"error": "need>=2_columns"}
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return [TableOut(name="KMeans", markdown="⚠️ 缺少 scikit-learn 依赖。")], [], {"error": "missing_sklearn"}

    X = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(X) < 10:
        return [TableOut(name="KMeans", markdown="⚠️ 有效样本不足（n<10）。")], [], {"error": "n_too_small"}
    k = int(max(2, min(int(k or 3), 10)))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(Xs)

    sizes = pd.Series(labels).value_counts().sort_index().reset_index()
    sizes.columns = ["cluster", "size"]

    centers = pd.DataFrame(km.cluster_centers_, columns=cols)
    centers.insert(0, "cluster", list(range(k)))

    # chart: PCA 2D
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.title("KMeans 聚类（PCA 2D 可视化）")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()

    tables = [
        TableOut(name="KMeans：簇大小", markdown=_safe_to_markdown(sizes, max_rows=20), df=sizes),
        TableOut(name="KMeans：中心（标准化空间）", markdown=_safe_to_markdown(centers, max_rows=30), df=centers),
    ]
    summary = {"columns": cols, "n": int(len(X)), "k": k, "sizes": sizes.to_dict(orient="records"), "inertia": float(km.inertia_)}
    return tables, [], summary


def analysis_logistic_regression(df: pd.DataFrame, *, y: str, x: List[str]) -> Tuple[List[TableOut], List[ChartOut], Dict[str, Any]]:
    if y not in df.columns:
        return [TableOut(name="Logistic回归", markdown="⚠️ y 列不存在。")], [], {"error": "missing_y"}
    x = [c for c in (x or []) if c in df.columns and c != y]
    if not x:
        return [TableOut(name="Logistic回归", markdown="⚠️ 需要至少 1 个自变量列。")], [], {"error": "missing_x"}

    y_raw = df[y]
    # 转为二元 0/1
    y_num = pd.to_numeric(y_raw, errors="coerce")
    if y_num.notna().sum() > 0 and set(y_num.dropna().unique().tolist()) <= {0, 1}:
        y_bin = y_num
        classes = [0, 1]
        mapping = None
    else:
        uniq = y_raw.dropna().astype(str).unique().tolist()
        if len(uniq) != 2:
            return [TableOut(name="Logistic回归", markdown="⚠️ y 需要是二元变量（两个类别或 0/1）。")], [], {"error": "y_not_binary", "unique": uniq[:10]}
        mapping = {uniq[0]: 0, uniq[1]: 1}
        y_bin = y_raw.astype(str).map(mapping)
        classes = [uniq[0], uniq[1]]

    X_df = df[x].apply(pd.to_numeric, errors="coerce")
    mask = ~(y_bin.isna() | X_df.isna().any(axis=1))
    y2 = y_bin[mask].astype(int)
    X2 = X_df[mask]
    n = int(len(y2))
    if n < 20:
        return [TableOut(name="Logistic回归", markdown="⚠️ 有效样本量不足（n<20）。")], [], {"error": "n_too_small"}

    # 尝试 statsmodels（提供 p 值/置信区间）
    try:
        import statsmodels.api as sm  # type: ignore

        X_sm = sm.add_constant(X2, has_constant="add")
        model = sm.Logit(y2, X_sm)
        res = model.fit(disp=0, maxiter=200)

        params = res.params
        pvals = res.pvalues
        ci = res.conf_int()
        ci.columns = ["ci_low", "ci_high"]

        rows = []
        for name in params.index:
            coef = float(params.loc[name])
            pv = float(pvals.loc[name]) if name in pvals.index else np.nan
            lo = float(ci.loc[name, "ci_low"]) if name in ci.index else np.nan
            hi = float(ci.loc[name, "ci_high"]) if name in ci.index else np.nan
            rows.append(
                {
                    "变量": str(name),
                    "系数": coef,
                    "OR": float(np.exp(coef)),
                    "p": pv,
                    "OR_95%CI": f"[{np.exp(lo):.4f},{np.exp(hi):.4f}]",
                }
            )
        coef_df = pd.DataFrame(rows)
        info_df = pd.DataFrame(
            [
                {
                    "y": y,
                    "x": ", ".join(x),
                    "n": n,
                    "LL": float(res.llf),
                    "AIC": float(res.aic),
                }
            ]
        )

        # ROC（可选）
        try:
            from sklearn.metrics import roc_auc_score, roc_curve
            import matplotlib.pyplot as plt

            prob = res.predict(X_sm)
            auc = float(roc_auc_score(y2, prob))
            fpr, tpr, _ = roc_curve(y2, prob)
            plt.figure(figsize=(5, 4))
            plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
            plt.plot([0, 1], [0, 1], "k--", linewidth=1)
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title("ROC 曲线")
            plt.legend()
            plt.tight_layout()
            chart_drawn = True
        except Exception:
            auc = None
            chart_drawn = False

        tables = [
            TableOut(name="Logistic回归：模型信息", markdown=_safe_to_markdown(info_df, max_rows=10), df=info_df),
            TableOut(name="Logistic回归：系数表(OR)", markdown=_safe_to_markdown(coef_df, max_rows=50), df=coef_df),
        ]
        summary = {"y": y, "x": x, "n": n, "classes": classes, "mapping": mapping, "coef": coef_df.to_dict(orient="records")}
        if auc is not None:
            summary["auc"] = auc
        summary["used"] = "statsmodels"
        return tables, [], summary

    except Exception as e:
        # fallback：sklearn（无 p 值）
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            from sklearn.metrics import roc_auc_score
        except Exception:
            return [TableOut(name="Logistic回归", markdown=f"⚠️ statsmodels 失败且缺少 sklearn：{e}")], [], {"error": "no_backend"}

        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
        pipe.fit(X2.values, y2.values)
        coef = pipe.named_steps["clf"].coef_.reshape(-1)
        intercept = float(pipe.named_steps["clf"].intercept_[0])
        rows = [{"变量": "截距", "系数": intercept, "OR": float(np.exp(intercept)), "p": np.nan}]
        for name, c in zip(x, coef):
            rows.append({"变量": name, "系数": float(c), "OR": float(np.exp(float(c))), "p": np.nan})
        coef_df = pd.DataFrame(rows)
        try:
            prob = pipe.predict_proba(X2.values)[:, 1]
            auc = float(roc_auc_score(y2.values, prob))
            # chart: ROC
            try:
                from sklearn.metrics import roc_curve
                import matplotlib.pyplot as plt

                fpr, tpr, _ = roc_curve(y2.values, prob)
                plt.figure(figsize=(5, 4))
                plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
                plt.plot([0, 1], [0, 1], "k--", linewidth=1)
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.title("ROC 曲线")
                plt.legend()
                plt.tight_layout()
            except Exception:
                pass
        except Exception:
            auc = None
        info_df = pd.DataFrame([{"y": y, "x": ", ".join(x), "n": n, "AUC": auc}])
        tables = [
            TableOut(name="Logistic回归：模型信息", markdown=_safe_to_markdown(info_df, max_rows=10), df=info_df),
            TableOut(name="Logistic回归：系数表(无p值)", markdown=_safe_to_markdown(coef_df, max_rows=50), df=coef_df),
        ]
        summary = {"y": y, "x": x, "n": n, "classes": classes, "mapping": mapping, "coef": coef_df.to_dict(orient="records"), "auc": auc, "used": "sklearn_fallback"}
        return tables, [], summary


def run_analysis(
    *, session_id: str, df: pd.DataFrame, analysis: str, params: Dict[str, Any], out_subdir: Optional[str] = None
) -> Dict[str, Any]:
    """
    统一入口：返回 {title,tables,charts,summary}
    charts 会在此处统一保存（因为 session_id 只在这里可用）。
    """
    analysis = (analysis or "").strip().lower()
    params = params or {}

    # 确保 out 目录存在
    _ensure_out_dir(session_id, out_subdir)

    import matplotlib.pyplot as plt

    plt.close("all")

    tables: List[TableOut] = []
    charts: List[ChartOut] = []
    summary: Dict[str, Any] = {}
    title = analysis

    if analysis in ("overview", "data_overview"):
        title = "数据概览"
        tables, _, summary = analysis_overview(df)
        chart_path = _save_current_fig(session_id, prefix="overview", out_subdir=out_subdir)
        if chart_path:
            charts.append(ChartOut(name="缺失率 Top15", path=chart_path))
        return {"title": title, "tables": tables, "charts": charts, "summary": summary}

    if analysis in ("frequency", "freq"):
        title = "频数分析"
        col = params.get("column")
        if not col or col not in df.columns:
            return {"title": title, "tables": [TableOut(name=title, markdown="⚠️ 请选择正确的列。")], "charts": [], "summary": {"error": "missing_column"}}
        tables, _, summary = analysis_frequency(df, column=col, top_n=int(params.get("top_n", 30)))
        # save chart drawn in analysis_frequency
        chart_path = _save_current_fig(session_id, prefix="freq", out_subdir=out_subdir)
        if chart_path:
            charts.append(ChartOut(name=f"频数图：{col}", path=chart_path))
        return {"title": title, "tables": tables, "charts": charts, "summary": summary}

    if analysis in ("crosstab", "contingency"):
        title = "列联(交叉)分析"
        row = params.get("row")
        col = params.get("col")
        if not row or not col or row not in df.columns or col not in df.columns:
            return {"title": title, "tables": [TableOut(name=title, markdown="⚠️ 请选择正确的 row/col 列。")], "charts": [], "summary": {"error": "missing_columns"}}
        ct, summ = analysis_crosstab(df, row=row, col=col)
        tables = [
            TableOut(name=f"列联表：{row} x {col}", markdown=_safe_to_markdown(ct.reset_index(), max_rows=80)),
            TableOut(name="卡方检验", markdown=_safe_to_markdown(pd.DataFrame([summ]), max_rows=10)),
        ]
        # heatmap
        import seaborn as sns

        plt.figure(figsize=(8, 5))
        sns.heatmap(ct, cmap="Blues")
        plt.title(f"列联热力图：{row} x {col}")
        plt.tight_layout()
        chart_path = _save_current_fig(session_id, prefix="crosstab", out_subdir=out_subdir)
        if chart_path:
            charts.append(ChartOut(name="列联热力图", path=chart_path))
        return {"title": title, "tables": tables, "charts": charts, "summary": summ}

    if analysis in ("descriptive", "describe"):
        title = "描述统计"
        cols = params.get("columns") or []
        if isinstance(cols, str):
            cols = [cols]
        tables, _, summary = analysis_descriptive(df, columns=cols)
        chart_path = _save_current_fig(session_id, prefix="describe", out_subdir=out_subdir)
        if chart_path:
            charts.append(ChartOut(name="分布箱线图", path=chart_path))
        return {"title": title, "tables": tables, "charts": charts, "summary": summary}

    if analysis in ("group_summary", "groupby"):
        title = "分类汇总"
        group_by = params.get("group_by")
        metric = params.get("metric")
        agg = params.get("agg", "mean")
        tables, _, summary = analysis_group_summary(df, group_by=str(group_by), metric=str(metric), agg=str(agg))
        chart_path = _save_current_fig(session_id, prefix="group", out_subdir=out_subdir)
        if chart_path:
            charts.append(ChartOut(name="分类汇总条形图", path=chart_path))
        return {"title": title, "tables": tables, "charts": charts, "summary": summary}

    if analysis in ("normality", "normal_test"):
        title = "正态性检验"
        col = params.get("column")
        method = params.get("method", "auto")
        if not col or col not in df.columns:
            return {"title": title, "tables": [TableOut(name=title, markdown="⚠️ 请选择正确的列。")], "charts": [], "summary": {"error": "missing_column"}}
        tables, _, summary = analysis_normality(df, column=col, method=str(method))
        chart_path = _save_current_fig(session_id, prefix="normality", out_subdir=out_subdir)
        if chart_path:
            charts.append(ChartOut(name="直方图+Q-Q图", path=chart_path))
        return {"title": title, "tables": tables, "charts": charts, "summary": summary}

    if analysis in ("ttest", "t_test"):
        title = "T检验"
        ttype = params.get("ttype", "independent")
        y = params.get("y")
        tables, _, summary = analysis_ttest(
            df,
            ttype=str(ttype),
            y=str(y),
            group_col=params.get("group_col"),
            group_a=params.get("group_a"),
            group_b=params.get("group_b"),
            y2=params.get("y2"),
            mu=float(params.get("mu", 0.0)),
        )
        # if a chart was created, save it
        chart_path = _save_current_fig(session_id, prefix="ttest", out_subdir=out_subdir)
        if chart_path:
            charts.append(ChartOut(name="图表", path=chart_path))
        return {"title": title, "tables": tables, "charts": charts, "summary": summary}

    if analysis in ("anova", "anova_oneway"):
        title = "单因素方差分析"
        y = params.get("y")
        group_col = params.get("group_col")
        tables, _, summary = analysis_anova_oneway(df, y=str(y), group_col=str(group_col))
        chart_path = _save_current_fig(session_id, prefix="anova", out_subdir=out_subdir)
        if chart_path:
            charts.append(ChartOut(name="箱线图", path=chart_path))
        return {"title": title, "tables": tables, "charts": charts, "summary": summary}

    if analysis in ("chi_square", "chi2"):
        title = "卡方检验"
        row = params.get("row")
        col = params.get("col")
        tables, _, summary = analysis_chi_square(df, row=str(row), col=str(col))
        chart_path = _save_current_fig(session_id, prefix="chi2", out_subdir=out_subdir)
        if chart_path:
            charts.append(ChartOut(name="热力图", path=chart_path))
        return {"title": title, "tables": tables, "charts": charts, "summary": summary}

    if analysis in ("nonparam", "non_param"):
        title = "非参数检验"
        test = params.get("test")
        tables, _, summary = analysis_nonparam(
            df,
            test=str(test),
            y=params.get("y"),
            group_col=params.get("group_col"),
            group_a=params.get("group_a"),
            group_b=params.get("group_b"),
            columns=params.get("columns"),
        )
        chart_path = _save_current_fig(session_id, prefix="nonparam", out_subdir=out_subdir)
        if chart_path:
            charts.append(ChartOut(name="图表", path=chart_path))
        return {"title": title, "tables": tables, "charts": charts, "summary": summary}

    if analysis in ("correlation", "corr"):
        title = "相关性分析"
        method = params.get("method", "pearson")
        cols = params.get("columns") or []
        if isinstance(cols, str):
            cols = [cols]
        tables, _, summary = analysis_correlation(df, method=str(method), columns=cols)
        chart_path = _save_current_fig(session_id, prefix="corr", out_subdir=out_subdir)
        if chart_path:
            charts.append(ChartOut(name="相关热力图", path=chart_path))
        return {"title": title, "tables": tables, "charts": charts, "summary": summary}

    if analysis in ("linear_regression", "linreg", "ols"):
        title = "线性回归"
        y = params.get("y")
        x = params.get("x") or []
        if isinstance(x, str):
            x = [x]
        tables, _, summary = analysis_linear_regression(df, y=str(y), x=list(x))
        chart_path = _save_current_fig(session_id, prefix="linreg", out_subdir=out_subdir)
        if chart_path:
            charts.append(ChartOut(name="预测 vs 实际", path=chart_path))
        return {"title": title, "tables": tables, "charts": charts, "summary": summary}

    if analysis in ("pca",):
        title = "主成分分析(PCA)"
        cols = params.get("columns") or []
        if isinstance(cols, str):
            cols = [cols]
        n_components = int(params.get("n_components", 2))
        tables, _, summary = analysis_pca(df, columns=list(cols), n_components=n_components)
        chart_path = _save_current_fig(session_id, prefix="pca", out_subdir=out_subdir)
        if chart_path:
            charts.append(ChartOut(name="PCA 图", path=chart_path))
        return {"title": title, "tables": tables, "charts": charts, "summary": summary}

    if analysis in ("kmeans", "k_means"):
        title = "聚类分析(K-Means)"
        cols = params.get("columns") or []
        if isinstance(cols, str):
            cols = [cols]
        k = int(params.get("k", 3))
        tables, _, summary = analysis_kmeans(df, columns=list(cols), k=k)
        chart_path = _save_current_fig(session_id, prefix="kmeans", out_subdir=out_subdir)
        if chart_path:
            charts.append(ChartOut(name="聚类可视化", path=chart_path))
        return {"title": title, "tables": tables, "charts": charts, "summary": summary}

    if analysis in ("logistic_regression", "logit"):
        title = "逻辑回归(Logistic)"
        y = params.get("y")
        x = params.get("x") or []
        if isinstance(x, str):
            x = [x]
        tables, _, summary = analysis_logistic_regression(df, y=str(y), x=list(x))
        chart_path = _save_current_fig(session_id, prefix="logit", out_subdir=out_subdir)
        if chart_path:
            charts.append(ChartOut(name="图表", path=chart_path))
        return {"title": title, "tables": tables, "charts": charts, "summary": summary}

    return {"title": analysis, "tables": [TableOut(name="未实现", markdown="⚠️ 该分析类型暂未实现。")], "charts": [], "summary": {"error": "unsupported_analysis"}}


