"""
流式响应测试
"""
import pytest
import json
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 现在可以正常导入
from backend.app.services.llm_service import LLMService


@pytest.mark.asyncio
async def test_llm_service_stream():
    """测试 LLM 服务的流式调用"""
    async with LLMService() as service:
        events = []
        async for event in service.call_llm_stream(
            provider="deepseekA",
            api_key="test-key",
            model_config={
                "deepseekA": {
                    "url": "https://api.deepseek.com/chat/completions",
                    "model": "deepseek-reasoner"
                }
            },
            prompt="Hello, world!",
        ):
            events.append(json.loads(event))
        
        # 验证事件格式
        assert len(events) > 0
        assert any(e.get("type") in ("content", "complete", "error") for e in events)


@pytest.mark.asyncio
async def test_llm_service_connection_pool():
    """测试连接池复用 - 在同一上下文中多个服务实例应该复用同一个连接池"""
    # 清理之前的连接池（如果有）
    if LLMService._client_pool is not None:
        await LLMService._client_pool.aclose()
        LLMService._client_pool = None
    
    # 测试：在同一上下文中，多个服务实例应该复用同一个连接池
    async with LLMService(use_pool=True) as service1:
        assert service1.client is not None
        client1_id = id(service1.client)
        
        # 在同一上下文中创建第二个服务实例
        async with LLMService(use_pool=True) as service2:
            assert service2.client is not None
            client2_id = id(service2.client)
            # 应该使用同一个客户端（连接池）
            assert client1_id == client2_id
            assert service1.client is service2.client


@pytest.mark.asyncio
async def test_llm_service_independent_client():
    """测试独立客户端（不使用连接池）"""
    async with LLMService(use_pool=False) as service:
        assert service.client is not None
        assert service._owns_client is True


def test_sse_format():
    """测试 SSE 格式"""
    event = {"type": "content", "content": "test"}
    sse_line = f"data: {json.dumps(event)}\n\n"
    
    assert sse_line.startswith("data: ")
    assert sse_line.endswith("\n\n")
    
    # 解析
    data_str = sse_line[6:].strip()
    parsed = json.loads(data_str)
    assert parsed == event


@pytest.mark.asyncio
async def test_stream_error_handling():
    """测试错误处理"""
    async with LLMService() as service:
        events = []
        async for event in service.call_llm_stream(
            provider="invalid_provider",
            api_key="",
            model_config={},
            prompt="test",
        ):
            events.append(json.loads(event))
        
        # 应该返回错误事件
        assert len(events) > 0
        assert any(e.get("type") == "error" for e in events)

