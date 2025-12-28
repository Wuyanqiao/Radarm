"""
工作流入口：生成数据分析报告（薄封装）

说明：
- 报告生成必须走“多专家混合-报告版”的独立工作流
- 具体引擎实现位于 `engine_report.py`
"""

from engine_report import run_report_engine


def run_workflow(user_request, data_context, api_keys, model_config, execute_callback, df):
    return run_report_engine(
        user_request=user_request,
        data_context=data_context,
        api_keys=api_keys,
        model_config=model_config,
        execute_callback=execute_callback,
        df=df,
    )