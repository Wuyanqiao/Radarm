"""
工作流入口：多专家混合 Agent（薄封装）

说明：
- `workflow_*` 只提供稳定的对外入口（供 backend.py 调用）
- 具体引擎实现位于 `engine_agent_multi.py`
"""

from engine_agent_multi import run_multi_agent_engine


def run_workflow(user_query, data_context, api_keys, model_config, execute_callback, df, roles=None):
    return run_multi_agent_engine(
        user_query=user_query,
        data_context=data_context,
        api_keys=api_keys,
        model_config=model_config,
        roles=roles,
        execute_callback=execute_callback,
        df=df,
    )