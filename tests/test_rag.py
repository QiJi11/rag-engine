"""
测试 rag.py 的检索逻辑
使用 mock 替代真实的 ChromaDB 和 CrossEncoder，避免网络/模型依赖
"""
import pytest
from unittest.mock import MagicMock, patch


class TestRetrieve:
    """测试两阶段检索函数 retrieve()"""

    @patch("services.rag.get_or_create_collection")
    def test_空集合返回空字符串(self, mock_get_col):
        """向量库为空时，retrieve 应直接返回空字符串"""
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_get_col.return_value = mock_col

        from services.rag import retrieve
        result = retrieve("随便问点啥")
        assert result == ""

    @patch("services.rag.get_or_create_collection")
    def test_少量文档跳过rerank(self, mock_get_col):
        """文档数 <= top_k 时，直接返回，不触发 CrossEncoder"""
        mock_col = MagicMock()
        mock_col.count.return_value = 2
        mock_col.query.return_value = {"documents": [["文档A", "文档B"]]}
        mock_get_col.return_value = mock_col

        with patch("services.rag.get_reranker") as mock_reranker:
            from services.rag import retrieve
            result = retrieve("问题", top_k=3)
            # 文档数(2) <= top_k(3)，不应调用 reranker
            mock_reranker.assert_not_called()
            assert "文档A" in result
            assert "文档B" in result

    @patch("services.rag.get_or_create_collection")
    @patch("services.rag.get_reranker")
    def test_多文档触发rerank(self, mock_get_reranker, mock_get_col):
        """文档数 > top_k 时，应触发 CrossEncoder rerank 并返回 top_k 个"""
        mock_col = MagicMock()
        mock_col.count.return_value = 5
        docs = ["文档一", "文档二", "文档三", "文档四", "文档五"]
        mock_col.query.return_value = {"documents": [docs]}
        mock_get_col.return_value = mock_col

        # 模拟 reranker：打分使"文档三"最高
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.1, 0.2, 0.9, 0.3, 0.4]
        mock_get_reranker.return_value = mock_reranker

        from services.rag import retrieve
        result = retrieve("测试查询", top_k=2)
        parts = result.split("\n---\n")
        assert len(parts) == 2
        # 最高分的"文档三"应排在第一位
        assert "文档三" in parts[0]

    @patch("services.rag.get_or_create_collection")
    def test_查询无结果返回空字符串(self, mock_get_col):
        """collection.query 返回空结果时应返回空字符串"""
        mock_col = MagicMock()
        mock_col.count.return_value = 3
        mock_col.query.return_value = {"documents": [[]]}
        mock_get_col.return_value = mock_col

        from services.rag import retrieve
        result = retrieve("无关查询")
        assert result == ""


class TestAddDocuments:
    """测试 add_documents 函数"""

    @patch("services.rag.get_or_create_collection")
    def test_正确调用collection_add(self, mock_get_col):
        """add_documents 应正确将 chunks 写入向量库"""
        mock_col = MagicMock()
        mock_get_col.return_value = mock_col

        chunks = [
            {"text": "第一段", "chunk_index": 0},
            {"text": "第二段", "chunk_index": 1},
        ]

        from services.rag import add_documents
        add_documents(chunks, "test_doc.pdf")

        mock_col.add.assert_called_once()
        call_kwargs = mock_col.add.call_args[1]
        assert len(call_kwargs["ids"]) == 2
        assert call_kwargs["ids"][0] == "test_doc.pdf_0"
        assert call_kwargs["ids"][1] == "test_doc.pdf_1"
        assert call_kwargs["documents"] == ["第一段", "第二段"]
