"""
代码执行引擎 - 支持 Plotly 和安全性增强
"""
import ast
import re
import sys
import time
import json
from io import StringIO
from pathlib import Path
from typing import Tuple, Optional, Any, Dict, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure

# 导入项目模块
# 注意：由于模块结构，需要从项目根目录导入
import sys
from pathlib import Path

# 添加项目根目录到路径（如果不在路径中）
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import ml_engine as ml


class CodeSecurityError(Exception):
    """代码安全检查失败异常"""
    pass


def _check_code_security(code_str: str) -> None:
    """
    使用 AST 检查代码安全性，禁止危险操作
    
    禁止的操作：
    - import os, sys, subprocess 等系统模块
    - 文件操作 (open, read, write)
    - 网络操作 (requests, urllib, socket)
    - 子进程操作 (subprocess, os.system)
    - 动态导入 (__import__, importlib)
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        raise CodeSecurityError(f"代码语法错误: {str(e)}")
    
    # 禁止的模块名
    FORBIDDEN_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'pathlib', 'tempfile',
        'requests', 'urllib', 'urllib2', 'urllib3', 'http', 'socket',
        'importlib', '__import__', 'eval', 'exec', 'compile',
        'ctypes', 'multiprocessing', 'threading', 'pickle', 'marshal'
    }
    
    # 禁止的函数调用
    FORBIDDEN_FUNCTIONS = {
        'open', 'file', 'input', 'raw_input', 'execfile',
        '__import__', 'eval', 'exec', 'compile', 'reload'
    }
    
    # 禁止的属性访问
    FORBIDDEN_ATTRS = {
        '__import__', '__builtins__', '__file__', '__name__', '__package__'
    }
    
    for node in ast.walk(tree):
        # 检查 import 语句
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split('.')[0]
                if module_name in FORBIDDEN_MODULES:
                    raise CodeSecurityError(
                        f"禁止导入模块: {module_name}。请直接使用已提供的库进行分析。"
                    )
        
        # 检查 from ... import 语句
        if isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module.split('.')[0]
                if module_name in FORBIDDEN_MODULES:
                    raise CodeSecurityError(
                        f"禁止从模块导入: {module_name}。请直接使用已提供的库进行分析。"
                    )
        
        # 检查函数调用
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_FUNCTIONS:
                    raise CodeSecurityError(
                        f"禁止调用函数: {node.func.id}。请使用已提供的工具函数。"
                    )
            elif isinstance(node.func, ast.Attribute):
                # 检查 os.system, subprocess.call 等
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id in FORBIDDEN_MODULES:
                        raise CodeSecurityError(
                            f"禁止调用: {node.func.value.id}.{node.func.attr}"
                        )
        
        # 检查属性访问（如 os.path, sys.path）
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                if node.value.id in FORBIDDEN_MODULES:
                    raise CodeSecurityError(
                        f"禁止访问: {node.value.id}.{node.attr}"
                    )
                if node.attr in FORBIDDEN_ATTRS:
                    raise CodeSecurityError(
                        f"禁止访问属性: {node.attr}"
                    )
        
        # 检查内置函数调用（如 eval, exec）
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ('eval', 'exec', 'compile', '__import__'):
                    raise CodeSecurityError(
                        f"禁止使用动态执行函数: {node.func.id}"
                    )


def execute_code(
    code_str: str, 
    df: pd.DataFrame, 
    session_id: str = "default"
) -> Tuple[str, Optional[str], Optional[str], pd.DataFrame]:
    """
    代码执行沙盒 - 支持 Plotly 和安全性增强
    
    参数:
        code_str: 要执行的 Python 代码字符串
        df: 数据框
        session_id: 会话 ID，用于保存图表
    
    返回:
        (output_text, image_path, plotly_json, new_df)
        - output_text: 执行输出文本
        - image_path: matplotlib 图表路径（相对路径，如 "session_id/filename.png"）
        - plotly_json: Plotly Figure 的 JSON 字符串（如果生成了 Plotly 图表）
        - new_df: 执行后的数据框
    """
    # 1. AST 安全检查
    try:
        _check_code_security(code_str)
    except CodeSecurityError as e:
        return str(e), None, None, df
    
    # 2. 正则表达式模式检查（作为额外安全层）
    forbidden_patterns = [
        r"\bopen\s*\(",
        r"\bpd\s*\.\s*read_(csv|excel|parquet|json)\s*\(",
        r"\bread_(csv|excel|parquet|json)\s*\(",
    ]
    for pat in forbidden_patterns:
        if re.search(pat, code_str, re.IGNORECASE):
            return (
                "禁止文件操作：请直接使用已提供的 df 进行分析（不要读取外部文件）。",
                None, None, df
            )
    
    # 3. 预置常用依赖
    local_vars: Dict[str, Any] = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "json": json,
        "plt": plt,
        "sns": sns,
        "ml": ml,
        "stats": stats,
        # Plotly 支持
        "px": px,
        "go": go,
        "plotly": __import__("plotly"),
    }
    
    # 4. 预置工具函数：列名模糊匹配
    def _norm_col_name(s: str) -> str:
        return re.sub(r"[\s\-_]+", "", str(s or "").strip().lower())
    
    def find_col(*candidates: str) -> Optional[str]:
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
    
    def list_cols() -> List[str]:
        _df = local_vars.get("df")
        return [str(c) for c in getattr(_df, "columns", [])] if _df is not None else []
    
    local_vars["find_col"] = find_col
    local_vars["list_cols"] = list_cols
    
    # 5. 回归模板工具函数
    def fit_linear_regression(y, X, feature_names=None):
        """
        拟合线性回归模型，返回系数、p值、R²、置信区间等。
        """
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
        
        # OLS 估计
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
        
        # 标准误与 t 统计量
        mse = ss_res / (n - k - 1) if n > k + 1 else ss_res / max(1, n - 1)
        try:
            var_beta = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            se_beta = np.sqrt(np.diag(var_beta))
            t_stats = beta / se_beta
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
    local_vars["_session_id"] = session_id
    
    # 6. 清空图表
    plt.clf()
    
    # 7. 执行代码
    try:
        # 捕获 print 输出
        old_stdout = sys.stdout
        redirected_output = StringIO()
        sys.stdout = redirected_output
        
        # 使用 compile + exec 而不是直接 exec（更安全）
        compiled_code = compile(code_str, '<string>', 'exec')
        exec(compiled_code, {}, local_vars)
        
        sys.stdout = old_stdout
        print_output = redirected_output.getvalue()
        
        text_res = str(local_vars.get('result', "执行成功"))
        
        # 优先使用 print 的内容作为硬结论
        final_output = print_output if print_output.strip() else text_res
        
        # 8. 处理 matplotlib 图表
        img_path = None
        if plt.get_fignums():
            session_id = local_vars.get("_session_id", "default")
            out_dir = Path("out") / session_id
            out_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time() * 1000)
            filename = f"chart_{timestamp}.png"
            filepath = out_dir / filename
            
            plt.savefig(str(filepath), format='png', bbox_inches='tight', dpi=300)
            plt.close()
            
            img_path = f"{session_id}/{filename}"
        
        # 9. 处理 Plotly 图表
        plotly_json = None
        # 检查 local_vars 中是否有 Plotly Figure 对象
        for var_name, var_value in local_vars.items():
            if isinstance(var_value, Figure):
                try:
                    plotly_json = var_value.to_json()
                    break
                except Exception as e:
                    # 如果转换失败，记录但不中断
                    final_output += f"\n[警告] Plotly 图表转换失败: {str(e)}"
        
        # 也检查是否有名为 'fig' 的变量（常见命名）
        if plotly_json is None and 'fig' in local_vars:
            fig = local_vars['fig']
            if isinstance(fig, Figure):
                try:
                    plotly_json = fig.to_json()
                except Exception as e:
                    final_output += f"\n[警告] Plotly 图表转换失败: {str(e)}"
        
        new_df = local_vars['df']
        return final_output, img_path, plotly_json, new_df
        
    except CodeSecurityError as e:
        sys.stdout = old_stdout
        return str(e), None, None, df
    except Exception as e:
        sys.stdout = old_stdout
        return f"Error: {str(e)}", None, None, df

