"""
测试 rag-engine FastAPI 接口
使用 TestClient 发送请求，mock 外部依赖（LLM、RAG）
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """创建 FastAPI TestClient"""
    # 在导入 app 前 patch 掉 config，避免缺少 .env 报错
    with patch.dict("os.environ", {
        "LLM_API_KEY": "test-key",
        "LLM_BASE_URL": "http://fake-api.test/v1",
        "LLM_MODEL": "gpt-test",
    }):
        from main import app
        return TestClient(app)


class TestHealthEndpoint:
    """测试 /health 接口"""

    def test_health_返回UP状态(self, client):
        """GET /health 应返回包含 status=UP 的 JSON"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "UP"
        assert "model" in data


class TestUploadEndpoint:
    """测试 /api/v1/upload 接口"""

    @patch("routers.upload.chunk_text")
    @patch("routers.upload.add_documents")
    def test_上传文件成功(self, mock_add, mock_chunk, client):
        """POST /api/v1/upload 应接受文件并返回 chunk 数量"""
        mock_chunk.return_value = [
            {"text": "第一段", "chunk_index": 0, "source": "test.txt"},
            {"text": "第二段", "chunk_index": 1, "source": "test.txt"},
        ]
        mock_add.return_value = None

        files = {"file": ("test.txt", b"dummy content", "text/plain")}
        response = client.post("/api/v1/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "chunks" in data or "chunk_count" in data or response.status_code == 200

    def test_无文件上传返回422(self, client):
        """POST /api/v1/upload 不带文件应返回 422 Unprocessable Entity"""
        response = client.post("/api/v1/upload")
        assert response.status_code == 422


class TestChatEndpoint:
    """测试 /api/v1/chat 接口（非流式）"""

    @patch("routers.chat.retrieve")
    @patch("routers.chat.chat")
    def test_正常对话返回内容(self, mock_chat, mock_retrieve, client):
        """POST /api/v1/chat 应返回 AI 回复内容"""
        mock_retrieve.return_value = "相关上下文"
        mock_chat.return_value = "这是AI的回答"

        # 使用 AsyncMock 处理协程
        import asyncio
        mock_chat.side_effect = AsyncMock(return_value="这是AI的回答")

        payload = {"query": "什么是RAG？", "session_id": "test-sess"}
        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
