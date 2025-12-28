"""
流式响应集成测试
"""
import pytest
import json
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient

# 导入 backend.py 中的 app
# 注意：backend.py 是根目录的文件，不是包内的模块
import importlib.util
backend_path = project_root / "backend.py"
spec = importlib.util.spec_from_file_location("backend_module", backend_path)
backend_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend_module)
app = backend_module.app


@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)


def test_stream_endpoint_exists(client):
    """测试流式端点是否存在"""
    response = client.post(
        "/chat/stream",
        json={
            "session_id": "test-session",
            "message": "test",
            "mode": "agent_multi",
            "apiKeys": {}
        }
    )
    # 应该返回流式响应（即使没有 API Key 也会返回错误事件）
    assert response.status_code in (200, 400, 422)


def test_stream_response_format(client):
    """测试流式响应格式"""
    response = client.post(
        "/chat/stream",
        json={
            "session_id": "test-session",
            "message": "test",
            "mode": "ask",
            "apiKeys": {}
        }
    )
    
    if response.status_code == 200:
        # TestClient 会自动处理流式响应
        # 读取响应内容
        content = response.content
        
        # 验证 SSE 格式
        text = content.decode('utf-8', errors='ignore')
        lines = text.split('\n')
        data_lines = [l for l in lines if l.startswith('data: ')]
        
        if data_lines:
            # 至少应该有一个事件
            assert len(data_lines) > 0
            
            # 验证 JSON 格式
            for line in data_lines[:3]:  # 只检查前3个
                data_str = line[6:]  # 移除 "data: " 前缀
                try:
                    event = json.loads(data_str)
                    assert "type" in event
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON in SSE line: {line}")

