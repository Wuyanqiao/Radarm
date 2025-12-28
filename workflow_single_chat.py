"""
工作流入口：单模型 Agent（薄封装）

说明：
- `workflow_*` 只提供稳定的对外入口（供 backend.py 调用）
- 具体引擎实现位于 `engine_agent_single.py`
"""

from engine_agent_single import run_single_agent_engine


def run_workflow(user_query, api_keys, primary_model, model_config, execute_callback, df, data_context: str = ""):
    return run_single_agent_engine(
        user_query=user_query,
        data_context=data_context,
        api_keys=api_keys,
        primary_model=primary_model,
        model_config=model_config,
        execute_callback=execute_callback,
        df=df,
    )