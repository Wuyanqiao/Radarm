#!/bin/bash
# 运行测试脚本

echo "运行流式响应测试..."
pytest tests/test_streaming.py -v

echo ""
echo "运行集成测试..."
pytest tests/test_integration_streaming.py -v

